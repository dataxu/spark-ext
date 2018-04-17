package org.apache.spark.ml.feature

import MultiTopKComputer.toMutable

/**
  * A TopKComputer that tracks top k values for n columns at the same time
  * @param columns the input columns to index
  * @param capacity the number of most frequent values to track
  * @param topKByColumn the topk elements per column being computed
  */
class MultiTopKComputer(
  val columns: Array[String],
  val capacity: Int,
  val topKByColumn: scala.collection.mutable.Map[String, TopKComputer])
  extends Serializable {

  def this(columns: Array[String], capacity: Int) =
    this(
      columns,
      capacity,
      toMutable(columns.map((_, new TopKComputer(capacity))).toMap))

  /**
    * adds a label to the topk computer of the column
    */
  def add(column: String, label: String): Unit = {
    topKByColumn.get(column) match {
      case Some(topKComputer: TopKComputer) => {
        topKComputer.add(label)
      }
      case None => throw invalidColumn(column)
    }
  }

  /**
    * @return the topk computer of a column
    */
  def get(column: String): TopKComputer = topKByColumn.get(column) match {
    case Some(topKComputer: TopKComputer) => topKComputer
    case None => throw invalidColumn(column)
  }

  /**
    * merges another top k computer into this one
    */
  def merge(other: MultiTopKComputer): Unit = {
    for (column <- columns) {
      (topKByColumn.get(column), other.topKByColumn.get(column)) match {
        case (Some(topKComputer1: TopKComputer), Some(topKComputer2: TopKComputer)) => {
          topKComputer1.merge(topKComputer2)
        }
        case _ => throw invalidColumn(column)
      }
    }
  }

  private def invalidColumn(column: String): IllegalArgumentException = {
    new IllegalArgumentException(s"$column does not exist in this topKComputer")
  }

  override def clone(): MultiTopKComputer = {
    val clonedTopKs = topKByColumn.map {
      case (column: String, topKComputer: TopKComputer) => {
        (column, topKComputer.clone())
      }
    }.toMap
    new MultiTopKComputer(
      columns,
      capacity,
      toMutable(clonedTopKs))
  }
}

object MultiTopKComputer {
  private def toMutable(topKByColumn: Map[String, TopKComputer]) = {
    scala.collection.mutable.Map[String, TopKComputer]() ++ topKByColumn
  }
}
