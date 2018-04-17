package dataxu.common.spark.testutil

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.StructType
import org.scalatest.{BeforeAndAfterAll, Suite}

/**
  * AbstractLocalSparkSession extension which does not have hive support enabled.
  */
trait LocalSparkSession extends AbstractLocalSparkSession { self: Suite =>
  override val enableHiveSupport: Boolean = false
}

/**
  * AbstractLocalSparkSession extension which has hive support enabled.
  */
trait LocalSparkSessionHiveEnabled extends AbstractLocalSparkSession { self: Suite =>
  override val enableHiveSupport: Boolean = true
}

/**
  * Trait that creates spark and sql context for unit tests. Use this trait when
  * there is no need to clean the spark context after each test.
  */
trait AbstractLocalSparkSession extends BeforeAndAfterAll { self: Suite =>
  var sparkSession: SparkSession = _
  var sc: SparkContext = _

  def enableHiveSupport: Boolean

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    val sessionBuilder = SparkSession.builder()
    if (enableHiveSupport) {
      sessionBuilder.enableHiveSupport()
    }
    sparkSession = sessionBuilder.config(getConf).getOrCreate()
    sc = sparkSession.sparkContext
  }

  override protected def afterAll(): Unit = {
    super.afterAll()
    resetSparkSession()
  }

  def resetSparkSession(): Unit = {
    LocalSparkSession.stop(sparkSession)
    sparkSession = null
    sc = null
  }

  def getConf: SparkConf = {
    val conf = new SparkConf(true)
    conf.setMaster("local[4]")
    conf.setAppName("unit-tests")
    conf.set("spark.ui.enabled", "false")
    // we need to set this so that unit tests can use multiple contexts.
    conf.set("spark.driver.allowMultipleContexts", "true")
    conf
  }

  protected def createDataFrame(rows: List[Row]): DataFrame = {
    // any of the rows contains the schema, so we just get the first row.
    val schema: StructType = rows.head.schema
    sparkSession.createDataFrame(sc.parallelize(rows), schema)
  }
}
object LocalSparkSession {
  def stop(sparkSession: SparkSession) {
    if (sparkSession != null) {
      sparkSession.stop()
    }
    // To avoid RPC rebinding to the same port, since it doesn't unbind immediately on shutdown
    System.clearProperty("spark.driver.port")
  }

  /** Runs `f` by passing in `sparkSession` and ensures that `sparkSession` is stopped. */
  def withSpark[T](sparkSession: SparkSession)(f: SparkSession => T): T = {
    try {
      f(sparkSession)
    } finally {
      stop(sparkSession)
    }
  }
}
