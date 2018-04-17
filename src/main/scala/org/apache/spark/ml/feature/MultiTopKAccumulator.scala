package org.apache.spark.ml.feature

import org.apache.spark.util.AccumulatorV2

/**
  * A Spark accumulator that is able to keep top k values for each column
  * @param columns the input columns to index
  * @param capacity the number of most frequent values to track
  */
class MultiTopKAccumulator(val columns: Array[String], val capacity: Int)
  extends AccumulatorV2[(String, String), MultiTopKComputer] {

  var multiTopK: Option[MultiTopKComputer] = None

  override def isZero: Boolean = multiTopK.isEmpty

  override def merge(other: AccumulatorV2[(String, String), MultiTopKComputer]): Unit = {
    multiTopK match {
      case None => {
        multiTopK = Some(other.value.clone())
      }
      case Some(stream) => {
        if (!other.isZero) {
          stream.merge(other.value)
        }
      }
    }
  }

  override def copy(): AccumulatorV2[(String, String), MultiTopKComputer] = {
    val accumulator = new MultiTopKAccumulator(columns, capacity)
    multiTopK match {
      case None => accumulator.multiTopK = None
      case Some(stream) => accumulator.multiTopK = Some(stream.clone())
    }
    accumulator
  }

  override def value: MultiTopKComputer = multiTopK match {
    case None => new MultiTopKComputer(columns, capacity)
    case Some(multiTopkComputer) => multiTopkComputer
  }

  override def add(columnAndLabel: (String, String)): Unit = {
    val (column, label) = columnAndLabel
    multiTopK match {
      case None => {
        val stream = new MultiTopKComputer(columns, capacity)
        stream.add(column, label)
        multiTopK = Some(stream)
      }
      case Some(stream) => {
        stream.add(column, label)
      }
    }
  }

  override def reset(): Unit = {
    multiTopK = None
  }

}
