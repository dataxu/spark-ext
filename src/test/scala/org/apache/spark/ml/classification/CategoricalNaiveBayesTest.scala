package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, VectorUDT}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.scalatest.{FunSpec, MustMatchers}
import org.scalatest.mockito.MockitoSugar

import dataxu.common.spark.testutil.{LocalSparkSession, DataFrameUtil, TempDir}

class CategoricalNaiveBayesTest extends FunSpec with LocalSparkSession
with MockitoSugar
with MustMatchers
with TempDir {

  val DataFrameStr =
    """
      +-------+------+-------+------+
      |country|region|zipcode| label|
      +-------+------+-------+------+
      |     US|    MA|     01|     0|
      |     US|    NH|     01|     1|
      |     UY|    MA|     01|     1|
      |     BR|    NH|     01|     0|
      |     BR|    MA|     01|     1|
      |     BR|    RI|     01|     0|
      |     US|    MA|     02|     0|
      |     US|    NH|     02|     1|
      |     UY|    MA|     02|     1|
      |     BR|    NH|     02|     1|
      |     BR|    MA|     02|     1|
      |     BR|    RI|     02|     0|
      +-------+------+-------+------+
    """
  lazy val baseDataSet = DataFrameUtil.fromShowFormat(
    sparkSession,
    "string,string,string,double",
    DataFrameStr
  )

  /**
    * conditional probabilities:
    *
    * // positives
    *
    * P(US|1) = 2/7
    * P(UY|1) = 2/7
    * P(BR|1) = 3/7
    *
    * P(NH|1) = 3/7
    * P(MA|1) = 4/7
    * P(RI|1) = 0
    *
    * P(01|1) = 3/7
    * P(02|1) = 4/7
    *
    *
    * // negatives
    *
    * P(US|0) = 2/5
    * P(UY|0) = 0
    * P(BR|0) = 3/5
    *
    * P(NH|0) = 1/5
    * P(MA|0) = 2/5
    * P(RI|0) = 2/5
    *
    * P(01|0) = 3/5
    * P(02|0) = 2/5
    *
    */

  describe("the naive bayes estimator") {
    it("obtains the expected predictions") {
      val (dataSet, model) = createDataFrameAndModel(20)
      dataSet.show()
      assertFirstRowCalculations(dataSet)
      val vectorHead = udf { x: DenseVector => f"${x(0)}%1.4f" }
      val vectorTail = udf { x: DenseVector => f"${x(1)}%1.4f" }
      val flattenedDataSet = dataSet
        .withColumn("negativeRawPrediction", vectorHead(dataSet("rawPrediction")))
        .withColumn("positiveRawPrediction", vectorTail(dataSet("rawPrediction")))
        .withColumn("negativeProbability", vectorHead(dataSet("rawPrediction")))
        .withColumn("positiveProbability", vectorTail(dataSet("rawPrediction")))
        .select(
          "country", "region", "zipcode",
          "negativeRawPrediction", "positiveRawPrediction",
          "negativeProbability", "positiveProbability")
      // scalastyle:off
      DataFrameUtil.assertDataFrame(flattenedDataSet,
        "string,string,string,string,string,string,string",
        """
          +-------+------+-------+---------------------+---------------------+-------------------+-------------------+
          |country|region|zipcode|negativeRawPrediction|positiveRawPrediction|negativeProbability|positiveProbability|
          +-------+------+-------+---------------------+---------------------+-------------------+-------------------+
          |     US|    MA|     01|               0.1050|               0.0823|             0.1050|             0.0823|
          |     US|    NH|     01|               0.0700|               0.0658|             0.0700|             0.0658|
          |     UY|    MA|     01|               0.2449|               0.0823|             0.2449|             0.0823|
          |     BR|    NH|     01|               0.0933|               0.0878|             0.0933|             0.0878|
          |     BR|    MA|     01|               0.1399|               0.1097|             0.1399|             0.1097|
          |     BR|    RI|     01|               0.1399|               0.1975|             0.1399|             0.1975|
          |     US|    MA|     02|               0.0787|               0.1029|             0.0787|             0.1029|
          |     US|    NH|     02|               0.0525|               0.0823|             0.0525|             0.0823|
          |     UY|    MA|     02|               0.1837|               0.1029|             0.1837|             0.1029|
          |     BR|    NH|     02|               0.0700|               0.1097|             0.0700|             0.1097|
          |     BR|    MA|     02|               0.1050|               0.1372|             0.1050|             0.1372|
          |     BR|    RI|     02|               0.1050|               0.2469|             0.1050|             0.2469|
          +-------+------+-------+---------------------+---------------------+-------------------+-------------------+
        """)
      // scalastyle:on
      computeAndAssertMissingValues(model)
    }


    def createDataFrameAndModel(
      k: Int,
      fields: Array[String] = Array("country", "region", "zipcode"))
    : (DataFrame, CategoricalNaiveBayesModel) = {
      val naiveBayes = new CategoricalNaiveBayes()
      naiveBayes.setInputCols(fields: _*)
      naiveBayes.setTopK(k)
      val model = naiveBayes.fit(baseDataSet)
      (model.transform(baseDataSet).select(col("*")), model)
    }

    it("transforms the schema by adding the predictor params") {
      val naiveBayes = new CategoricalNaiveBayes()
      naiveBayes.setInputCols("country", "region")
      naiveBayes.setTopK(20)
      val transformedSchema = naiveBayes.transformSchema(baseDataSet.schema)
      transformedSchema must be(new StructType(Array(
        StructField("country", StringType, true),
        StructField("region", StringType, true),
        StructField("zipcode", StringType, true),
        StructField("label", DoubleType, true),
        StructField("features", new VectorUDT(), false),
        StructField("rawPrediction", new VectorUDT(), false),
        StructField("probability", new VectorUDT(), false),
        StructField("prediction", DoubleType, false))))
    }
  }

  def assertFirstRowCalculations(df: DataFrame): Unit = {
    val rawNegativeProd = 3.0 / 7.0 * 3.0 / 7.0 * 4.0 / 7.0 // values are smoothed
    val rawPositiveProd = 3.0 / 9.0 * 5.0 / 9.0 * 4.0 / 9.0 // values are smoothed
    val rows = df.take(1)
    rows(0).getAs[DenseVector](5).apply(0) must be(rawNegativeProd +- 0.0001)
    rows(0).getAs[DenseVector](5).apply(1) must be(rawPositiveProd +- 0.0001)
  }

  def computeAndAssertMissingValues(model: CategoricalNaiveBayesModel): Unit = {
    val missingValuesDataFrameStr =
      """
        +-------+------+-------+
        |country|region|zipcode|
        +-------+------+-------+
        |     US|    MA|     99|
        +-------+------+-------+
      """
    val missingValuesDataFrame = DataFrameUtil.fromShowFormat(
      sparkSession,
      "string,string,string",
      missingValuesDataFrameStr
    )
    val predictionsWithMissingValues = model.transform(missingValuesDataFrame)
    predictionsWithMissingValues.show()
    assertMissingValuesCalculations(predictionsWithMissingValues)
  }

  def assertMissingValuesCalculations(df: DataFrame): Unit = {
    val rawNegativeProd = 3.0 / 7.0 * 3.0 / 7.0 // values are smoothed
    val rawPositiveProd = 3.0 / 9.0 * 5.0 / 9.0 // values are smoothed
    val rows = df.collect()
    rows(0).getAs[DenseVector](4).apply(0) must be(rawNegativeProd +- 0.0001)
    rows(0).getAs[DenseVector](4).apply(1) must be(rawPositiveProd +- 0.0001)
  }
}
