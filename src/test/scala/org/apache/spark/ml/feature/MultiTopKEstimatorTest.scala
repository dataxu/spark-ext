package org.apache.ml.feature

import java.io.File

import org.apache.spark.ml.feature.{MultiTopKEstimator, MultiTopKModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, Metadata}
import org.scalatest.{FunSpec, MustMatchers}
import org.scalatest.mockito.MockitoSugar

import dataxu.common.spark.testutil.{LocalSparkSession, DataFrameUtil, TempDir}

class MultiTopKEstimatorTest
  extends FunSpec
  with LocalSparkSession
  with MockitoSugar
  with MustMatchers
  with TempDir {

  val DataFrameStr =
    """
      +-------+------+-------+
      |country|region|zipcode|
      +-------+------+-------+
      |     US|    MA|     23|
      |     US|    NH|     12|
      |     UY|    MA|     23|
      |     BR|    NH|     12|
      |     BR|    MA|     12|
      |     BR|    RI|      9|
      +-------+------+-------+
    """

  lazy val dataFrame = DataFrameUtil.fromShowFormat(
    sparkSession,
    "string,string,int",
    DataFrameStr
  )

  describe("the topk estimator") {
    it("indexes the expected features values for k = 2") {
      val actualDf = createDataFrame(2)
      actualDf.show()
      // scalastyle:off
      DataFrameUtil.assertDataFrame(actualDf,
        "string,string,int,int",
        """
          +-------+------+-------+-------------+
          |country|region|zipcode|country_index|
          +-------+------+-------+-------------+
          |     US|    MA|     23|            2|
          |     US|    NH|     12|            2|
          |     UY|    MA|     23|            0|
          |     BR|    NH|     12|            1|
          |     BR|    MA|     12|            1|
          |     BR|    RI|      9|            1|
          +-------+------+-------+-------------+
        """
      )
      // scalastyle:on
    }

    it("indexes the expected features values for k = 3") {
      val actualDf = createDataFrame(3)
      DataFrameUtil.assertDataFrame(actualDf,
        "string,string,int,int",
        """
          +-------+------+-------+-------------+
          |country|region|zipcode|country_index|
          +-------+------+-------+-------------+
          |     US|    MA|     23|            2|
          |     US|    NH|     12|            2|
          |     UY|    MA|     23|            3|
          |     BR|    NH|     12|            1|
          |     BR|    MA|     12|            1|
          |     BR|    RI|      9|            1|
          +-------+------+-------+-------------+
        """
      )
      actualDf.schema("country_index").metadata must be(
        Metadata.fromJson(
          """
            |{
            |    "ml_attr": {
            |        "name": "country_index",
            |        "type": "nominal",
            |        "vals": [
            |            "/m",
            |            "BR",
            |            "US",
            |            "UY"
            |        ]
            |    }
            |}
            |
          """.stripMargin)
      )
    }

    it("indexes the expected region features values for k = 2") {
      val actualDf = createDataFrame(2, Array("region"))
      DataFrameUtil.assertDataFrame(actualDf,
        "string,string,int,int",
        """
          +-------+------+-------+------------+
          |country|region|zipcode|region_index|
          +-------+------+-------+------------+
          |     US|    MA|     23|           1|
          |     US|    NH|     12|           2|
          |     UY|    MA|     23|           1|
          |     BR|    NH|     12|           2|
          |     BR|    MA|     12|           1|
          |     BR|    RI|      9|           0|
          +-------+------+-------+------------+
        """
      )
    }

    it("indexes the expected numerical features values for k = 2") {
      val actualDf = createDataFrame(2, Array("zipcode"))
      DataFrameUtil.assertDataFrame(actualDf,
        "string,string,int,int",
        """
          +-------+------+-------+-------------+
          |country|region|zipcode|zipcode_index|
          +-------+------+-------+-------------+
          |     US|    MA|     23|            2|
          |     US|    NH|     12|            1|
          |     UY|    MA|     23|            2|
          |     BR|    NH|     12|            1|
          |     BR|    MA|     12|            1|
          |     BR|    RI|      9|            0|
          +-------+------+-------+-------------+
        """
      )
    }

    it("indexes multiple features values for k = 2") {
      val actualDf = createDataFrame(2, Array("country", "region", "zipcode"))
      DataFrameUtil.assertDataFrame(actualDf,
        "string,string,int,int,int,int",
        """
         +-------+------+-------+-------------+------------+-------------+
          |country|region|zipcode|country_index|region_index|zipcode_index|
         +-------+------+-------+-------------+------------+-------------+
          |     US|    MA|     23|            2|           1|            2|
          |     US|    NH|     12|            2|           2|            1|
          |     UY|    MA|     23|            0|           1|            2|
          |     BR|    NH|     12|            1|           2|            1|
          |     BR|    MA|     12|            1|           1|            1|
          |     BR|    RI|      9|            1|           0|            0|
         +-------+------+-------+-------------+------------+-------------+
        """
      )
    }

    it("transforms the schema correctly") {
      val (transformedDataframe, model) =
        createDataFrameAndModel(2, Array("country", "region", "zipcode"))
      val newSchema = model.transformSchema(dataFrame.schema)
      newSchema.size must be(dataFrame.schema.size + 3)
      newSchema("country_index").dataType must be(DoubleType)
      newSchema("region_index").dataType must be(DoubleType)
      newSchema("zipcode_index").dataType must be(DoubleType)
      newSchema("zipcode_index").metadata must be(Metadata.fromJson(
        """
          {
              "ml_attr": {
                  "type": "nominal"
              }
          }
        """))
    }

    it("is successfully serialized") {
      val model = trainModel(dataFrame, Array("zipcode"), 2)
      val modelFile = new File(tmpDir, "topk")
      model.save(modelFile.getAbsolutePath)
      val loadedModel = MultiTopKModel.load(modelFile.getAbsolutePath)

      model.uid must be(loadedModel.uid)
    }

    def trainModel(dataFrame: DataFrame, fields: Array[String], k: Int): MultiTopKModel = {
      val topk = new MultiTopKEstimator()
      topk.setInputCols(fields)
      topk.setOutputCols(fields.map(_ + "_index"))
      topk.setTopK(k)
      topk.fit(dataFrame)
    }

    def createDataFrameAndModel(k: Int, fields: Array[String] = Array("country"))
    : (DataFrame, MultiTopKModel) = {
      val model = trainModel(dataFrame, fields, k)
      (model.transform(dataFrame).select(col("*")), model)
    }

    def createDataFrame(k: Int, fields: Array[String] = Array("country")): DataFrame =
      trainModel(dataFrame, fields, k).transform(dataFrame)
  }
}
