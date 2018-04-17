package org.apache.spark.ml.feature

import scala.collection.JavaConverters._

import com.clearspring.analytics.stream.{Counter, StreamSummary}

/**
  * A probabilistic datastructure that is able to track the top k most frequent
  * features using O(k) memory space.
  */
class TopKComputer(val capacity: Int) extends Serializable {

  private val streamSummary = new StreamSummary[String](capacity * TopKComputer.CapacityFactor)
  private var totalCounter = 0L

  /**
    * records a new value
    * @param value the value to record
    * @param count the amount of times this value is occurring
    */
  def add(value: String, count: Int = 1): Unit = {
    streamSummary.offer(value, count)
    totalCounter += count
  }

  /**
    * calculates the topk values
    * @return the most frequent top k values
    */
  def values: Array[String] =
    streamSummary.topK(capacity).iterator().asScala.map {
      case counter: Counter[String] => counter.getItem
    }.toArray

  /**
    * calculates the topk values and their corresponding counts
    * @return the most frequent top k values
    */
  def valueCounts: Map[String, Long] =
    streamSummary.topK(capacity).iterator().asScala.map {
      case counter: Counter[String] => (counter.getItem, counter.getCount)
    }.toMap

  /**
    * merges another top k instance to this one
    * @param topKComputer
    */
  def merge(topKComputer: TopKComputer): Unit = {
    val counters = topKComputer.streamSummary.topK(topKComputer.capacity)
    for (i <- 0 until counters.size()) {
      streamSummary.offer(
        counters.get(i).getItem,
        counters.get(i).getCount.toInt)
    }
    totalCounter += topKComputer.totalCount
  }

  override def clone(): TopKComputer = {
    val cloned = new TopKComputer(capacity)
    cloned.merge(this)
    cloned
  }

  /**
    * @return the total amount of elements added to this counter
    */
  def totalCount: Long = totalCounter

}

object TopKComputer {
  private val CapacityFactor = 2
}
