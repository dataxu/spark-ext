package dataxu.common.spark.testutil

import java.io.File

import com.google.common.io.Files
import org.apache.commons.io.FileUtils
import org.scalatest.{BeforeAndAfterEach, FunSpec}

/**
  * Trait that provides the creation and deletion of a temporary directory for testing.
  */
trait TempDir extends FunSpec with BeforeAndAfterEach {

  protected var tmpDir: File = _

  override protected def beforeEach(): Unit = {
    super.beforeEach()
    tmpDir = Files.createTempDir()
  }

  override protected def afterEach(): Unit = {
    super.afterEach()
    FileUtils.deleteDirectory(tmpDir)
  }

}
