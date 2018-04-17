package org.apache.spark.ml.classification

import org.apache.spark.annotation.{DeveloperApi, Since}
import org.apache.spark.ml.{linalg, Estimator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, VectorUDT}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * An estimator that is able to create a Naive Bayes model for categorical features.
  *
  * The input columns of this classifier can be strings as opposed to many other Spark classifiers.
  * This implementation will store conditional probabilities for the top k most frequent nominals
  * for each feature. Other values will be treated as missing values upon prediction. Simple laplace
  * smoothing is used for conditional probabilities.
  *
  * Sample usage:
  *
  * val naiveBayes = new CategoricalNaiveBayes()
  * naiveBayes.setInputCols("feature1", "feature2")
  * naiveBayes.setTopK(200)
  * naiveBayes.setLabelCol("label")
  * val model = naiveBayes.fit(trainDf)
  *
  * @param uid the pipeline uid
  */
class CategoricalNaiveBayes(override val uid: String)
  extends Estimator[CategoricalNaiveBayesModel]
  with DefaultParamsWritable
  with ProbabilisticClassifierParams
  with HasInputCols {

  def this() = this(Identifiable.randomUID("categoricalNaiveBayes"))

  final val topK: Param[Int] = new Param[Int](this, "topk", "topk value")

  /**
    * Sets column names for input features
    *
    * @param columns the column names for input features
    * @return this instance
    */
  def setInputCols(columns: Array[String]): this.type = set(inputCols, columns)

  /**
    * Sets column names for input features as varargs
    *
    * @param columns the column names for input features
    * @return this instance
    */
  def setInputCols(columns: String*): this.type = setInputCols(columns.toArray)

  /**
    * Sets the top k value. This classifier will keep at most k most frequent nominals
    * for each feature. The rest will be considererd missing values.
    *
    * @param value the top k value.
    * @return this instance
    */
  def setTopK(value: Int): this.type = set(topK, value)

  setDefault(topK, MultiTopKEstimator.DefaultTopKValue)

  /**
    * Set the name of the label column
    * @param value the label column name
    * @return this classifier
    */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /**
    * @return the configured top k value.
    */
  def getTopK: Int = $(topK)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): CategoricalNaiveBayesModel = {
    logParams(dataset)
    val positivesAccumulator = new MultiTopKAccumulator($(inputCols), getTopK)
    val negativesAccumulator = new MultiTopKAccumulator($(inputCols), getTopK)
    dataset.sparkSession.sparkContext.register(negativesAccumulator, "neg_accum")
    dataset.sparkSession.sparkContext.register(positivesAccumulator, "pos_accum")
    dataset.asInstanceOf[DataFrame].foreach { row: Row =>
      $(inputCols).foreach { column: String =>
        val columnAndValue = (column, row.getAs(column).toString)
        if (row.getAs[Double]($(labelCol)) < 1.0) {
          negativesAccumulator.add(columnAndValue)
        } else {
          positivesAccumulator.add(columnAndValue)
        }
      }
    }
    val positiveCondProbabilities = conditionalProbabilities(positivesAccumulator.value)
    val negativeCondProbabilities = conditionalProbabilities(negativesAccumulator.value)
    positivesAccumulator.reset()
    negativesAccumulator.reset()
    val modelCopy = new CategoricalNaiveBayesModel(
      uid,
      positiveCondProbabilities,
      negativeCondProbabilities).setParent(this)
    copyValues(modelCopy)
  }

  /**
    * computes the laplace smoothed conditional probabilites for each nominal for each feature
    * feature name -> nominal -> P(F = x | L = y)
    */
  private def conditionalProbabilities(multiTopKComputer: MultiTopKComputer) =
    multiTopKComputer.topKByColumn.map {
      case (column: String, topKComputer: TopKComputer) =>
        (column, topKComputer.valueCounts.map {
          case (nominal, count) =>
            (nominal, (count + 1) / (topKComputer.totalCount.toDouble + 2))
        })
    }.toMap

  override def copy(extra: ParamMap): Estimator[CategoricalNaiveBayesModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(parentSchema: StructType): StructType = {
    List(
      ($(featuresCol), new VectorUDT),
      ($(rawPredictionCol), new VectorUDT),
      ($(probabilityCol), new VectorUDT),
      ($(predictionCol), DoubleType)
    ).foldLeft(parentSchema) {
      case (aggSchema: StructType, (feature: String, dataType: DataType)) =>
        SchemaUtils.appendColumn(aggSchema, feature, dataType)
    }
  }

  private def logParams(dataset: Dataset[_]): Unit = {
    val instr = Instrumentation.create(this, dataset)
    instr.logParams(
      labelCol, featuresCol, predictionCol, rawPredictionCol, probabilityCol, thresholds)
  }

}

/**
  * A model for CategoricalNaiveBayes
  * @param uid the model uid
  * @param positiveRatios the smoothed nominal conditional probabilities for positive instances
  *                       feature name -> nominal -> P(F = x | L = 1)
  * @param negativeRatios the smoothed nominal conditional probabilities for negative instances
  *                       feature name -> nominal -> P(F = x | L = 0)
  */
case class CategoricalNaiveBayesModel(
    uid: String,
    positiveRatios: Map[String, Map[String, Double]],
    negativeRatios: Map[String, Map[String, Double]])
  extends ProbabilisticClassificationModel[Seq[String], CategoricalNaiveBayesModel]
  with MLWritable {

  val inputFeatureNamesSeq = negativeRatios.keySet.toSeq

  override def copy(extra: ParamMap): CategoricalNaiveBayesModel = {
    val copied = new CategoricalNaiveBayesModel(uid, positiveRatios, negativeRatios)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = throw new UnsupportedOperationException("writing not supported")

  override protected def raw2probabilityInPlace(rawPrediction: linalg.Vector): linalg.Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        val probSum = dv.values.sum
        dv.values(0) = dv(0) / probSum
        dv.values(1) = dv(1) / probSum
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in NaiveBayesModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val inputFeatureColumns = inputFeatureNamesSeq.map(col(_))

    val assembled = dataset.select(
      List[Column](
        col("*"),
        array(inputFeatureColumns: _*).as($(featuresCol))): _*)
    super.transform(assembled)
  }

  protected override def validateAndTransformSchema(schema: StructType,
                                                    fitting: Boolean,
                                                    featuresDataType: DataType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }

  override protected def predictRaw(features: Seq[String]): linalg.Vector = {

    // calculate the product of all conditional probabilities using the log trick
    def probabilityProduct(
      features: Seq[String],
      conditionalProbabilities: Map[String, Map[String, Double]]) =
      math.exp(inputFeatureNamesSeq.zip(features).map {
        case (featureName: String, featureValue: String) =>
          featureConditionalProbability(featureName, featureValue, conditionalProbabilities)
      }.filter(_ != MultiTopKModel.MissingValueLabel).filter {
        d => !d.isNaN && !d.isInfinity
      }.sum)

    // calculate one feature value conditional probability
    def featureConditionalProbability(
      featureName: String,
      featureValue: String,
      conditionalProbabilities: Map[String, Map[String, Double]]) =
      conditionalProbabilities.get(featureName) match {
        case Some(probability) => math.log(probability.getOrElse(featureValue, 0.0))
        case None => throw new IllegalStateException(s"feature $featureName does not exist")
      }

    new DenseVector(
      Array(
        probabilityProduct(features, negativeRatios),
        probabilityProduct(features, positiveRatios)))
  }

  override def numClasses: Int = 2
}
