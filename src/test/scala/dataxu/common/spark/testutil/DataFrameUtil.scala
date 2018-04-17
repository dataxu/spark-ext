package dataxu.common.spark.testutil

import java.sql.Date
import java.text.SimpleDateFormat

import com.typesafe.scalalogging.slf4j.LazyLogging
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
/**
  * Test utilities for working with dataframes.
  */
object DataFrameUtil extends LazyLogging {

  val NullValue = "null"
  val CommaEscapeCharacter = """\\"""
  val supportedDateFormat = new SimpleDateFormat("yyyy-MM-dd")

  // auxiliary data types
  val nullableStringArrayDataType = new ArrayType(StringType, true)
  val nullableIntArrayDataType = new ArrayType(IntegerType, true)
  val nullableStringLongMapDataType = new MapType(StringType, LongType, true)
  val nullableStringDoubleMapDataType = new MapType(StringType, DoubleType, true)
  val nullableStringStringMapDataType = new MapType(StringType, StringType, true)
  val nullableStringStringMapArrayDataType = new ArrayType(nullableStringStringMapDataType, true)

  /**
    * Creates a dataframe from a csv-style string.
    *
    * @param sparkSession a Spark Session
    * @param str the string representing the dataframe
    *            Example:
                  country:string,region:string,zipcode:int,date:date
                  US,MA,22,2016-08-21
                  US,NH,11,2016-08-21
                  UY,MV,23,2016-08-21
                  BR,SA,12,2016-08-22
                  BR,GU,12,2016-08-22
                  BR,RI,9,2016-08-22
    * @return the parsed dataframe.
    */
  def fromString(sparkSession: SparkSession, str: String): DataFrame = {
    val lines = str.split("\n").filter { s: String => s.size > 1 }
    val headers: String = lines(0)
    val schema = StructType(headers.split(",").map {
      case (fieldAndType: String) => {
        val splits = fieldAndType.trim().split(":")
        val (name: String, fieldType: String) = (splits(0), splits(1))
        StructField(name, parseDataType(fieldType), nullable = true)
      }
    }.toList)
    val rows = lines.slice(1, lines.size).map(parseRow(_, schema))
    val sc = sparkSession.sparkContext
    sparkSession.createDataFrame(sc.parallelize(rows), schema)
  }


  /**
    * Creates a dataframe from a spark-show style string.
    *
    * @param sparkSession a Spark Session
    * @param types  a comma separated list of the types of each column i.e. "string,int,double,date"
    * @param str the string representing the dataframe
    *            Example:
                  +-------+------+-------+----------+
                  |country|region|zipcode|      date|
                  +-------+------+-------+----------+
                  |     US|    MA|     23|2016-08-21|
                  |     US|    NH|     12|2016-08-21|
                  |     UY|    MA|     23|2016-08-21|
                  |     BR|    NH|     12|2016-08-22|
                  |     BR|    MA|     12|2016-08-22|
                  |     BR|    RI|      9|2016-08-22|
                  +-------+------+-------+----------|
    * @return the parsed dataframe.
    */
  def fromShowFormat(sparkSession: SparkSession, types: String, str: String) : DataFrame = {
    var cleaned = str.trim().replaceAll("(\\+-)|(--)|(-\\+)", "").replace('|', ',').split("\n")
    cleaned = cleaned.filter { s: String => s.size > 1 }
    cleaned = cleaned.map(_.trim())
    cleaned = cleaned.map({ case x: String =>
      if (x.endsWith(",")) x.substring(0, x.size - 1) else x
    })
    cleaned = cleaned.map({ case x: String => if (x.startsWith(",")) x.substring(1) else x })
    cleaned = cleaned.filter { s: String => s.size > 1 }

    val zipped = cleaned(0).split(",").zip(types.split(","))
    val headers = zipped.map({ case (x: String, y: String) => x + ":" + y }).mkString(",").trim()
    val prepared = (headers :: cleaned.slice(1, cleaned.size).toList).mkString("\n")
    fromString(sparkSession, prepared)
  }

  private def parseDataType(fieldType: String): DataType = {
    fieldType.trim().toLowerCase() match {
      case "string" => StringType
      case "byte" => ByteType
      case "int" => IntegerType
      case "long" => LongType
      case "double" => DoubleType
      case "boolean" => BooleanType
      case "date" => DateType
      case "array_string" => nullableStringArrayDataType
      case "array_int" => nullableIntArrayDataType
      case "map_string_long" => nullableStringLongMapDataType
      case "map_string_double" => nullableStringDoubleMapDataType
      case "map_string_string" => nullableStringStringMapDataType
      case "array_map_string_string" => nullableStringStringMapArrayDataType
    }
  }

  private def parseRow(rowStr: String, schema: StructType): Row = {
    // We allow input row string to have escaped commas as \\, which are then replaced for a ,
    val splits = rowStr.split(s"""(?<!$CommaEscapeCharacter),""")
                       .map(split => split.replace(s"""$CommaEscapeCharacter,""", ","))
    val fields = for (i <- 0 until splits.size)
      yield parseField(splits(i).trim(), schema.fields(i).dataType)
    Row.fromSeq(fields.toList)
  }

  private def parseField(value: String, dataType: DataType): Any = {
    if (value.equals(NullValue)) return null

    dataType match {
      case StringType => value
      case ByteType => value.toByte
      case IntegerType => value.toInt
      case LongType => value.toLong
      case DoubleType => value.toDouble
      case BooleanType => value.toBoolean
      case DateType => stringToDate(value)
      case `nullableStringArrayDataType` => value.split(";")
      case `nullableIntArrayDataType` => value.split(";").map(_.toInt)
      case `nullableStringLongMapDataType` => stringToMapStringLong(value)
      case `nullableStringDoubleMapDataType` => stringToMapStringDouble(value)
      case `nullableStringStringMapDataType` => stringToMapStringString(value)
      case `nullableStringStringMapArrayDataType` =>
        value.split("\\^").map(stringToMapStringString(_))
    }
  }

  private def stringToDate(dateAsString: String): Date = {
    new Date(supportedDateFormat.parse(dateAsString).getTime)
  }

  private def stringToMapStringString(mapAsString: String,
                                      splitCharacter: String = ";"): Map[String, String] =
    Map[String, String](mapAsString.split(splitCharacter)
                        .flatMap(value => equalSignSplit(value)): _*)

  private def equalSignSplit(value: String) = {
    if (value.contains("=")) {
      val split = value.split("=")
      Some(split(0), split(1))
    } else {
      None
    }
  }

  private def stringToMapStringLong(mapAsString: String): Map[String, Long] =
  // The .map(identity) at the end is due to a known bug in Scala SI-7005
  // where .mapValues is not serializable
    stringToMapStringString(mapAsString).mapValues[Long](_.toLong).map(identity)

  private def stringToMapStringDouble(mapAsString: String): Map[String, Double] =
  // The .map(identity) at the end is due to a known bug in Scala SI-7005
  // where .mapValues is not serializable
    stringToMapStringString(mapAsString).mapValues[Double](_.toDouble).map(identity)

  /**
    * Asserts that a dataframe is equal to a show-format representation of spark, regardless the
    * order of their rows.
    *
    * @param actualDataFrame the data frame we want to assert.
    * @param types the types of the columns as comma-separated i.e. "string,int,double".
    * @param stringDataFrame the string representation of the expected dataframe (in show-format).
    */
  def assertDataFrameOrderInsensitive(actualDataFrame: DataFrame,
                                      types: String,
                                      stringDataFrame: String): Unit = {
    val expectedDf = fromShowFormat(actualDataFrame.sparkSession, types, stringDataFrame)
    assertDataFrameOrderInsensitive(expectedDf, actualDataFrame)
  }

  /**
    * Compares two data frames and ensures they are equal despite the rows are in the same order or
    * not.
    *
    * @param expectedDf the expected data frame.
    * @param actualDf the actual data frame.
    */
  def assertDataFrameOrderInsensitive(expectedDf: DataFrame, actualDf: DataFrame): Unit = {
    val sortedExpectedDataFrame = expectedDf.sort(getColumnSeqFromDataframe(expectedDf): _*)
    val sortedActualDataFrame = actualDf.sort(getColumnSeqFromDataframe(actualDf): _*)
    assertDataFrame(sortedExpectedDataFrame, sortedActualDataFrame)
  }

  /**
    * Auxiliary function to get a Seq[Column] of all the columns in the data frame.
    * @param dataFrame the data frame we want to get all the sequence of its columns.
    * @return a sequence of all the columns of the data frame.
    */
  private def getColumnSeqFromDataframe(dataFrame: DataFrame): Seq[org.apache.spark.sql.Column] = {
    dataFrame.columns.map(dataFrame.col)
  }

  /**
    * Compares two dataframes and ensures they are equal.
    *
    * @param expectedDf the expected data frame.
    * @param actualDf the actual data frame.
    */
  def assertDataFrame(expectedDf: DataFrame, actualDf: DataFrame): Unit = {
    val rows1 = expectedDf.collect().toSeq
    val rows2 = actualDf.collect().toSeq
    logger.debug(s"expectedDf: $rows1.toString)")
    logger.debug(s"actualDf: $rows2.toString)")
    assert(rows1.size == rows2.size,
      s"dataframes are not the same size:${rows1.size} vs ${rows2.size}")
    rows1.zip(rows2) foreach {
      case (row1: Row, row2: Row) => {
        assert(row1.equals(row2), s"rows are not equal. expected:\n $row1 \n actual: \n $row2")
      }
    }
  }

  /**
    * Asserts that a dataframe is equal to a show-format representation of spark.
    *
    * @param actualDataFrame the data frame we want to test.
    * @param types the types of the columns as comma-separated i.e. "string,int,double".
    * @param stringDataFrame the string representation of the expected dataframe (in show-format).
    */
  def assertDataFrame(actualDataFrame: DataFrame, types: String, stringDataFrame: String): Unit = {
    val expectedDf = fromShowFormat(actualDataFrame.sparkSession, types, stringDataFrame)
    assertDataFrame(expectedDf, actualDataFrame)
  }

  /**
    * Escapes all the commas in a given string using the comma escape character.
    *
    * @param string the string in which to escape the commas.
    * @return the string with the commas escaped.
    */
  def escapeCommasInString(string: String): String =
    string.replace(",", s"""$CommaEscapeCharacter,""")

}
