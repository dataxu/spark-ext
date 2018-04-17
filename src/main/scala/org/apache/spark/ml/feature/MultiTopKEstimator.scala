package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWritable, MLWriter, _}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * Represents a Spark Estimator that performs a similar transformation
  * to that of the StringIndexer. However this estimator will only use the top K
  * most frequent feature values and assign indices in the range [1,K].
  * The rest of the feature values will map to the value 0. Additionally this estimator
  * will be able to compute all top k features in 1 pass.
  */
class MultiTopKEstimator(override val uid: String)
  extends Estimator[MultiTopKModel]
  with DefaultParamsWritable
  with MultiTopKBase {

  def this() = this(Identifiable.randomUID("multiTopKIdx"))

  final val topK: Param[Int] = new Param[Int](this, "topk", "topk value")

  /** @group setParam */
  def setInputCols(columns: Array[String]): this.type = set(inputCols, columns)

  /** @group setParam */
  def setOutputCols(columns: Array[String]): this.type = set(outputCols, columns)

  /**
    * Sets the top k value.
    *
    * @param value the topk value.
    * @return this instance with the top k value set.
    */
  def setTopK(value: Int): this.type = set(topK, value)

  setDefault(topK, MultiTopKEstimator.DefaultTopKValue)

  /**
    * @return the configured top k value.
    */
  def getTopK: Int = $(topK)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): MultiTopKModel = {
    val topKAccumulator = new MultiTopKAccumulator($(inputCols), getTopK)
    dataset.sparkSession.sparkContext.register(topKAccumulator, "topkacc")
    dataset match {
      case dataFrame: DataFrame => {
        dataFrame.foreach {
          r: Row => {
            $(inputCols).foreach { column: String =>
              val columnValue = r.getAs[Any](column)
              val columnStringValue =
                if (columnValue == null) MultiTopKModel.MissingValueLabel else columnValue.toString
              topKAccumulator.add((column, columnStringValue))
            }
          }
        }
        val topKMappings = topKAccumulator.value.topKByColumn.map {
          case (column: String, topKComputer: TopKComputer) => (column, topKComputer.values)
        }.toMap
        topKAccumulator.reset()
        copyValues(new MultiTopKModel(uid, topKMappings).setParent(this))
      }
      case _ => throw new IllegalArgumentException("dataset needs to be of type Row")
    }
  }

  override def copy(extra: ParamMap): Estimator[MultiTopKModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

/**
  * MultiTopK trait common methods and traits shared by the estimator and the model
  */
trait MultiTopKBase extends Params with HasInputCols with HasOutputCols {

  def validateAndTransformSchema(schema: StructType): StructType = {
    val inputFields = schema.fields
    val outputFields = $(inputCols).zip($(outputCols)) flatMap {
      case (inputColName: String, outputColName: String) =>
        val inputDataType = schema(inputColName).dataType
        require(inputDataType == StringType || inputDataType.isInstanceOf[NumericType],
          s"The input column $inputColName must be either string type or numeric type, " +
            s"but got $inputDataType.")
        require(inputFields.forall(_.name != outputColName),
          s"Output column $outputColName already exists.")
        val attr = NominalAttribute.defaultAttr.withName(outputColName)
        Some(attr.toStructField())
    }
    StructType(inputFields ++ outputFields)
  }
}

/**
  * MultiTopKModel is capable of indexing a set of features given the top k labels per feature
  * @param uid the model uid
  * @param topKModels the top k labels per feature
  */
case class MultiTopKModel(val uid: String, topKModels: Map[String, Array[String]])
  extends Model[MultiTopKModel] with MLWritable with MultiTopKBase {

  val columnLabelToIndex = topKModels.toList.map {
    case (column: String, labels: Array[String]) =>
      (column, (for (i <- labels.indices) yield labels(i) -> (i + 1)).toMap)
  }.toMap

  val indexer = udf { (inputCol: String, label: String) =>
    indexLabel(inputCol, label)
  }

  /**
    * @param label the label to index (i.e. the nominal)
    * @return the numerical representation using topk clamping
    */
  def indexLabel(column: String, label: Any): Int = {
    if (label == null) {
      MultiTopKModel.MissingValueIndex
    } else {
      columnLabelToIndex.get(column) match {
        case Some(labelMap) => labelMap.getOrElse(label.toString, MultiTopKModel.MissingValueIndex)
        case None => {
          throw new IllegalArgumentException(s"invalid column $column")
        }
      }
    }
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    val columns = List(col("*")) ++ $(inputCols).zip($(outputCols)).map {
      case (inputCol: String, outputCol: String) =>
        val metadata = NominalAttribute.defaultAttr
          .withName(outputCol)
          .withValues(MultiTopKModel.MissingValueLabel +: topKModels(inputCol)).toMetadata()
        indexer(lit(inputCol), col(inputCol).cast(StringType)).as(outputCol, metadata)
    }
    dataset.select(columns: _*)
  }

  override def copy(extra: ParamMap): MultiTopKModel = {
    val copied = new MultiTopKModel(uid, topKModels)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new MultiTopKModelWriter(this)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object MultiTopKEstimator {
  val DefaultTopKValue = 500
}

object MultiTopKModel extends MLReadable[MultiTopKModel] {
  /**
    * The index of a missing value.
    */
  final val MissingValueIndex = 0
  /**
    * The label of missing value.
    */
  final val MissingValueLabel = "/m"

  override def read: MLReader[MultiTopKModel] = new MultiTopKModelReader
}

private class MultiTopKModelWriter(instance: MultiTopKModel) extends MLWriter {

  private case class Data(column: String, labels: Array[String])

  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
    val data = instance.topKModels.map {
      case (column: String, labels: Array[String]) => Data(column, labels)
    }.toSeq
    val dataPath = new Path(path, "data").toString
    sparkSession.createDataFrame(data).repartition(1).write.parquet(dataPath)
  }
}

private class MultiTopKModelReader extends MLReader[MultiTopKModel] {

  private val className = classOf[MultiTopKModel].getName

  override def load(path: String): MultiTopKModel = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
    val dataPath = new Path(path, "data").toString
    val data = sparkSession.read.parquet(dataPath).select("column", "labels").collect()
    val topKModels = data.map { r: Row =>
      (r.getAs[String]("column"), r.getAs[Seq[String]]("labels").toArray)
    }
    val model = new MultiTopKModel(metadata.uid, topKModels.toList.toMap)
    DefaultParamsReader.getAndSetParams(model, metadata)
    model
  }
}

/**
  * Trait for shared param inputCols.
  */
private[ml] trait HasOutputCols extends Params {

  /**
    * Param for output column names.
    * @group param
    */
  final val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "output column names")

  /** @group getParam */
  final def getOutputCols: Array[String] = $(outputCols)
}
