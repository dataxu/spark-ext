package org.apache.spark.ml.feature

import org.scalatest.{FunSpec, MustMatchers}

class TopKComputerTest extends FunSpec with MustMatchers {

  describe("the top k computer") {
    it("initializes correctly") {
      val topKComputer = new TopKComputer(3)
      topKComputer.values.size must be(0)
      topKComputer.totalCount must be (0)

    }
    it("calculates correctly the top k values") {
      val topKComputer = new TopKComputer(3)
      topKComputer.add("A", 3)
      topKComputer.add("B", 2)
      topKComputer.add("C")
      topKComputer.add("D")
      topKComputer.values must be(Array("A", "B", "C"))
      topKComputer.totalCount must be (7)
    }
    it("merges correctly another computer") {
      val topKComputer = new TopKComputer(3)
      topKComputer.add("A", 3)
      topKComputer.add("B", 2)
      val otherComputer = new TopKComputer(3)
      otherComputer.add("C", 1)
      otherComputer.add("D", 1)
      topKComputer.merge(otherComputer)
      topKComputer.values must be(Array("A", "B", "C"))
    }
  }
}
