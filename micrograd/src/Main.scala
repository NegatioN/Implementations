object Main extends App {
  case class Value(value: Float, children: Set[Value] = Set.empty) {
    var grad = 0.0f
    val prev: Set[Value] = children
    var _backward: () => Unit = () => {}

    def *(that: Value): Value = {
      val result = Value(value * that.value, Set(this, that))
      def _backward() = {
        this.grad += result.grad * that.value
        that.grad += result.grad * this.value
      }
      result._backward = _backward
      result
    }

    def +(that: Value): Value = {
      val result = Value(value + that.value, Set(this, that))
      def _backward() = {
        this.grad += result.grad
        that.grad += result.grad
      }
      result._backward = _backward
      result
    }

    def -(that: Value): Value = this.+(that * Value(-1.0f)) // This creates new nodes in the graph, but will suffice

    def backward = {
      grad = 1.0f
      def backwardRec(values: Set[Value]): Unit = {
        values.foreach { value =>
          value._backward()
          backwardRec(value.prev)
        }
      }
      backwardRec(Set(this))
  }

    override def toString: String = f"Value($value%.2f, grad=$grad%.2f)"
  }

  val a = Value(5.0f)
  val b = Value(3.0f)
  val c = Value(3.0f)

  val d = a + b
  val e = d * c
  e.backward

  println(a, b, c, d, e)
}