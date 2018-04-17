package org.apache.spark.ml.feature

import org.scalatest.{FunSpec, MustMatchers}

class MultiTopKComputerTest extends FunSpec with MustMatchers {

  describe("the MultiTopKComputer ") {
    it("initializes correctly") {
      val computer = new MultiTopKComputer(Array("col1", "col2"), 3)
      computer.get("col1").values.size must be(0)
      computer.get("col2").values.size must be(0)
    }
    it("adds items correctly") {
      val computer = sampleComputer
      computer.get("col1").values must be(Array("A", "B"))
      computer.get("col2").values must be(Array("A", "C"))
    }
    it("fails upon attempt to add bad column") {
      val computer = sampleComputer
      assertThrows[IllegalArgumentException] {
        computer.add("invalidCol", "A")
      }
    }
    it("merges another computer correctly") {
      val computer = sampleComputer
      val anotherComputer = new MultiTopKComputer(Array("col1", "col2"), 2)
      anotherComputer.add("col2", "D")
      anotherComputer.add("col2", "D")
      anotherComputer.add("col1", "X")
      anotherComputer.add("col1", "X")
      computer.merge(anotherComputer)
      computer.get("col1").values must be(Array("X", "A"))
      computer.get("col2").values must be(Array("D", "A"))
    }
    it("clones correctly") {
      val computer = sampleComputer
      val clone = sampleComputer.clone()
      clone.add("col1", "X")
      clone.add("col1", "X")
      computer.get("col1").values must be(Array("A", "B"))
      computer.get("col2").values must be(Array("A", "C"))
      clone.get("col1").values must be(Array("X", "A"))
      clone.get("col2").values must be(Array("A", "C"))
    }
  }

  private def sampleComputer = {
    val computer = new MultiTopKComputer(Array("col1", "col2"), 2)
    computer.add("col1", "A")
    computer.add("col1", "B")
    computer.add("col1", "D")
    computer.add("col2", "A")
    computer.add("col2", "C")
    computer
  }

}
