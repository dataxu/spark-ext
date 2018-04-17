package org.apache.spark.ml.feature

import org.scalatest.{FunSpec, MustMatchers}

class MultiTopKAccumulatorTest extends FunSpec with MustMatchers {

  val columns = Array("col1", "col2")

  describe("the top k accumulator") {
    it("works successfully correctly when adding from scratch") {
      val accumulator = new MultiTopKAccumulator(columns, 4)
      accumulator.isZero must be(true)
      accumulator.add(("col1", "A"))
      accumulator.isZero == false
      accumulator.add(("col1", "A"))
      accumulator.add(("col1", "B"))
      accumulator.add(("col2", "C"))
      accumulator.add(("col2", "C"))
      accumulator.value.get("col1").values must be(Array("A", "B"))
      accumulator.value.get("col2").values must be(Array("C"))

    }

    it("merges two full accumulators successfully") {
      val acc1 = new MultiTopKAccumulator(columns, 4)
      val acc2 = new MultiTopKAccumulator(columns, 4)
      for (i <- 1.to(5)) {
        acc1.add(("col1", s"$i"))
      }
      for (i <- 3.to(5)) {
        acc2.add(("col1", s"$i"))
      }
      acc1.merge(acc2)
      acc1.value.get("col1").values must be(Array("3", "4", "5", "1"))
    }

    it("merges two accumulators successfully, first one empty") {
      val acc1 = new MultiTopKAccumulator(columns, 4)
      val acc2 = new MultiTopKAccumulator(columns, 4)
      for (i <- 3.to(5)) {
        acc2.add(("col1", s"$i"))
      }
      acc1.merge(acc2)
      acc1.value.get("col1").values must be(Array("3", "4", "5"))
    }

    it("merges two accumulators successfully, last one empty") {
      val acc1 = new MultiTopKAccumulator(columns, 4)
      val acc2 = new MultiTopKAccumulator(columns, 4)
      for (i <- 3.to(5)) {
        acc2.add(("col1", s"$i"))
      }
      acc2.merge(acc1)
      acc2.value.get("col1").values must be(Array("3", "4", "5"))
    }

    it("copies one accumulator correctly") {
      val accumulator = new MultiTopKAccumulator(columns, 4)
      accumulator.add(("col1", "A"))
      accumulator.add(("col1", "A"))
      accumulator.add(("col1", "B"))
      val copied = accumulator.copy()
      copied.value.get("col1").values must be(Array("A", "B"))
      copied.add(("col1", "B"))
      copied.add(("col1", "B"))
      copied.add(("col1", "B"))
      accumulator.value.get("col1").values must be(Array("A", "B"))
      copied.value.get("col1").values must be(Array("B", "A"))
    }

    it("correctly resets the accumulators") {
      val accumulator = new MultiTopKAccumulator(columns, 4)
      accumulator.add(("col1", "A"))
      accumulator.reset()
      accumulator.isZero must be(true)
    }
  }


}
