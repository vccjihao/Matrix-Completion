/* This code is for testing a rank revealing randomized algorithm for large scale
* matrix completion problems

* We acknowledge support from RSCA (Research, Scholarship, and Creative
* Activities Program) 2016-2017 at Cal Poly Pomona*/


import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.random.StandardNormalGenerator
import org.apache.spark.mllib.linalg.{Vector, Vectors,Matrices}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, axpy => brzAxpy, svd => brzSvd, MatrixSingularException, inv}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.github.fommil.netlib.ARPACK
import org.netlib.util.{intW, doubleW}
import java.io.PrintWriter
import breeze.linalg._
import breeze.math._
import breeze.numerics._
import java.io._

object MatrixCompletion_randomized_rrsvd {

	var flag_1e4 = false;
	var flag_1e5 = false;
	var flag_1e6 = false;
	var flag_1e7 = false;
	var flag_1e8 = false;
	var flag_1e9 = false;


	val EPSILON = {
		var eps = 1.0
		while ((1.0 + (eps / 2.0)) != 1.0) {
		  eps /= 2.0
		}
		eps
	}

	/* display a graph*/
	def printGraph(g: Graph[Array[Double], Double]) = {
		val facts: RDD[String] = g.triplets.map(triplet => triplet.srcId + "("+ triplet.srcAttr+ ")" + " --- " + triplet.attr + " --- " + triplet.dstId + "("+ triplet.dstAttr+ ")")
		facts.collect().foreach(println)
	}

	/* display vertices in a graph*/
	def printSortedVerticesInGraph(g: Graph[Array[Double], Double]) = {
		g.vertices.sortBy(_._1).collect()
	}

	def printSortedVertices(v: VertexRDD[Array[Double]]) = {
		v.sortBy(_._1).collect()
	}

	def loadMat(sc: SparkContext, filename: String, blocksize: Int): Graph[Array[Double],Double] = {
		val graphFile = sc.textFile(filename)
		val data = graphFile.map{
			line => val parts = line.split(' ') 
			(parts(0).toDouble.toLong, parts(1).toDouble.toLong, parts(2).toDouble) 
		}
		
		val edges = data.flatMap{
			case (i,j,s) => 
			if(s<0.0){
				throw new SparkException("Similarity must be nonnegative but found s($i,$j) = $s")
			}
			Seq(Edge(i,j,s))
		}
		val rawG = Graph.fromEdges(edges, 0.0)

		val vD = rawG.aggregateMessages[Double](
			sendMsg = ctx => {ctx.sendToDst(ctx.attr)},
			mergeMsg = _ + _,
			TripletFields.EdgeOnly
		)
		val rawG2 = Graph(vD,rawG.edges).mapTriplets( e => e.attr / math.max(e.dstAttr, EPSILON), new TripletFields(false,true,true))
		
		Graph.fromEdges(rawG2.edges, Array.fill(blocksize)(0.0))
	}


	/* generates random values in vertices*/
	def randomInit(g: Graph[Array[Double], Double], blocksize: Int): Graph[Array[Double], Double] = {
		val r = g.vertices.mapPartitionsWithIndex(
			(partId,iter) => {
				val random = new StandardNormalGenerator()
				random.setSeed(partId)
				iter.map{ case (id,_) => (id, Array.fill(blocksize)(random.nextValue))}
			}, preservesPartitioning = true
		).cache()
		
		val sum = r.values.map(e=> e.map(math.abs)).reduce(_.zip(_).map{case (x,y) => x+y})
		val v0 = r.mapValues(x => x.zip(sum).map{case (x,y) => x/y})
		
		Graph(VertexRDD(v0), g.edges)
	}

	def output(g: Graph[Array[Double], Double], file: String): Unit  = {
		val pw = new PrintWriter(new File(file))
		val b = g.vertices.sortBy(_._1).collect
		val n = g.vertices.count - 1
		val nj = b(0)._2.length -1
		
		var i = 0
		var j = 0
		for(i <- 0 to n.toInt){
			var temp = b(i)._1.toString
			val a : Array[Double]= b(i)._2;
			for(j <- 0 to nj.toInt)
			{
				temp = temp + " " + a(j).toString
			}
			pw.println(temp)
		}
		pw.close()
    }


	/* randomized rrsvd for matrix completion*/
	def randomized_rrsvd(sc: SparkContext, g: Graph[Array[Double], Double], blocksize: Int, maxIters: Int, checkpointInterval: Int, tol: Double): VertexRDD[Array[Double]] ={
		sc.setCheckpointDir("/user/hxxji001/temp")
		val cp = new PeriodicGraphCheckpointer[Array[Double], Double](checkpointInterval, sc)
		var iter = 1;
		
		var gi = g;
		var diffDelta = 10000.00;
		for (iter <- 1 to maxIters if math.abs(diffDelta) > tol){
			val v = gi.aggregateMessages[Array[Double]](
				sendMsg = ctx => ctx.sendToSrc(ctx.dstAttr.map(_*ctx.attr)),
				mergeMsg = _.zip(_).map{case (x,y) => x+y},
				new TripletFields(false, true, true)).cache()

			gi = Graph(VertexRDD(v), g.edges);

			cp.update(gi)
			
			if(iter % checkpointInterval == 0 || iter == maxIters){
				diffDelta = test(v,g,iter);
			}
			/*
			materialize(gi)*/		
			v.unpersist(blocking = false)

		}
		gi.vertices
	}

	 def materialize(g: Graph[_, _]): Unit = {
		g.vertices.count()
		g.edges.count()
	  } 
  
	/*
	* Compute the smallest k eigenvalues and eigenvectors on a symmetric square matrix using ARPACK
	* slightly modify the EigenValueDecomposition from 
	https://github.com/apache/spark/blob/v1.6.1/mllib/src/main/scala/org/apache/spark/mllib/linalg/EigenValueDecomposition.scala
	*/
	def symmetricEigs(
	  mul: BDV[Double] => BDV[Double],
	  n: Int,
	  k: Int,
	  tol: Double,
	  maxIterations: Int): (BDV[Double], BDM[Double]) = {
	// TODO: remove this function and use eigs in breeze when switching breeze version
	require(n > k, s"Number of required eigenvalues $k must be smaller than matrix dimension $n")

	val arpack = ARPACK.getInstance()

	// tolerance used in stopping criterion
	val tolW = new doubleW(tol)
	// number of desired eigenvalues, 0 < nev < n
	val nev = new intW(k)
	// nev Lanczos vectors are generated in the first iteration
	// ncv-nev Lanczos vectors are generated in each subsequent iteration
	// ncv must be smaller than n
	val ncv = math.min(2 * k, n)

	// "I" for standard eigenvalue problem, "G" for generalized eigenvalue problem
	val bmat = "I"
	// "SM" : compute the NEV smallest (in magnitude) eigenvalues
	val which = "SM"

	var iparam = new Array[Int](11)
	// use exact shift in each iteration
	iparam(0) = 1
	// maximum number of Arnoldi update iterations, or the actual number of iterations on output
	iparam(2) = maxIterations
	// Mode 1: A*x = lambda*x, A symmetric
	iparam(6) = 1

	require(n * ncv.toLong <= Integer.MAX_VALUE && ncv * (ncv.toLong + 8) <= Integer.MAX_VALUE,
	  s"k = $k and/or n = $n are too large to compute an eigendecomposition")

	var ido = new intW(0)
	var info = new intW(0)
	var resid = new Array[Double](n)
	var v = new Array[Double](n * ncv)
	var workd = new Array[Double](n * 3)
	var workl = new Array[Double](ncv * (ncv + 8))
	var ipntr = new Array[Int](11)

	// call ARPACK's reverse communication, first iteration with ido = 0
	arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr, workd,
	  workl, workl.length, info)

	val w = BDV(workd)

	// ido = 99 : done flag in reverse communication
	while (ido.`val` != 99) {
	  if (ido.`val` != -1 && ido.`val` != 1) {
		throw new IllegalStateException("ARPACK returns ido = " + ido.`val` +
			" This flag is not compatible with Mode 1: A*x = lambda*x, A symmetric.")
	  }
	  // multiply working vector with the matrix
	  val inputOffset = ipntr(0) - 1
	  val outputOffset = ipntr(1) - 1
	  val x = w.slice(inputOffset, inputOffset + n)
	  val y = w.slice(outputOffset, outputOffset + n)
	  y := mul(x)
	  // call ARPACK's reverse communication
	  arpack.dsaupd(ido, bmat, n, which, nev.`val`, tolW, resid, ncv, v, n, iparam, ipntr,
		workd, workl, workl.length, info)
	}

	if (info.`val` != 0) {
	  info.`val` match {
		case 1 => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
			" Maximum number of iterations taken. (Refer ARPACK user guide for details)")
		case 3 => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
			" No shifts could be applied. Try to increase NCV. " +
			"(Refer ARPACK user guide for details)")
		case _ => throw new IllegalStateException("ARPACK returns non-zero info = " + info.`val` +
			" Please refer ARPACK user guide for error message.")
	  }
	}

	val d = new Array[Double](nev.`val`)
	val select = new Array[Boolean](ncv)
	// copy the Ritz vectors
	val z = java.util.Arrays.copyOfRange(v, 0, nev.`val` * n)

	// call ARPACK's post-processing for eigenvectors
	arpack.dseupd(true, "A", select, d, z, n, 0.0, bmat, n, which, nev, tol, resid, ncv, v, n,
	  iparam, ipntr, workd, workl, workl.length, info)

	// number of computed eigenvalues, might be smaller than k
	val computed = iparam(4)

	val eigenPairs = java.util.Arrays.copyOfRange(d, 0, computed).zipWithIndex.map { r =>
	  (r._1, java.util.Arrays.copyOfRange(z, r._2 * n, r._2 * n + n))
	}

	// sort the eigen-pairs in descending order
	val sortedEigenPairs = eigenPairs.sortBy(- _._1)

	// copy eigenvectors in descending order of eigenvalues
	val sortedU = BDM.zeros[Double](n, computed)
	sortedEigenPairs.zipWithIndex.foreach { r =>
	  val b = r._2 * n
	  var i = 0
	  while (i < n) {
		sortedU.data(b + i) = r._1._2(i)
		i += 1
	  }
	}

	(BDV[Double](sortedEigenPairs.map(_._1)), sortedU)
	}


  def outputVector(g: Graph[_, _], file: String): Unit  = {
  	val pw = new PrintWriter(new File(file))
	val b = g.vertices.sortBy(_._1).collect
	val n = g.vertices.count - 1
	
	var i = 0
	for(i <- 0 to n.toInt){
		pw.println(b(i)._1 + " " +b(i)._2)
	}
	pw.close()
  }	
	
	
	
  def test(v:VertexRDD[Array[Double]], g: Graph[Array[Double], Double], iter: Int ) : Double = {
		/* Convert RDD[Array]to RDD[IndexedRow]*/
		val v2 = v.map(e => IndexedRow(e._1.toLong, Vectors.dense( e._2)))

		/* Construct IndexedRowMatrix from RDD[IndexedRow]*/
		val Qk = new IndexedRowMatrix(v2)

		/* Zk2 = Qk-P*Qk */
		val r = Qk.rows.map( e  => (e.index, e.vector.toArray))
		val g3 = Graph(VertexRDD(r),g.edges)
		val zk = g3.aggregateMessages[Array[Double]](
					sendMsg = ctx => ctx.sendToSrc(ctx.dstAttr.map(_*ctx.attr*(-1))),
					mergeMsg = _.zip(_).map{case (x,y) => x+y},
					new TripletFields(false, true, true)).cache()
		val zk2 = zk.map(e => IndexedRow(e._1.toLong, Vectors.dense( e._2)))
		val zk3 = new IndexedRowMatrix(zk2)
		val Zk = zk3.toBlockMatrix();

		val Qk2 = Qk.toBlockMatrix();
		val Zk2 = Qk2.add(Zk);

		/* Bk = Zk'*Zk */
		val ZkT = Zk2.transpose
		val Bk = ZkT.multiply(Zk2).toLocalMatrix()

		/* eig(Bk)*/
		val breezeMatrix = new BDM[Double](Bk.numCols, Bk.numRows, Bk.toArray)
		val sym = breezeMatrix.asInstanceOf[BDM[Double]]

		val eig_result = eigSym(sym)
		val eig_vec = eig_result.eigenvectors
		val idx = eig_result.eigenvalues.argmin
		val eig_vec2 = eig_vec(::,idx)
		
		val n = Bk.numCols.toInt
		val eigvec = Matrices.dense(n,1,eig_vec2.toArray)
		val eigvec2 = Qk.multiply(eigvec)

		/*norm(Pv-v)*/
		val r2 = eigvec2.rows.map( e  => (e.index, e.vector.apply(0)))
		val g4: Graph[Double,Double]= Graph(VertexRDD(r2),g.edges)

		val r3 = g4.aggregateMessages[Double]( sendMsg = ctx => ctx.sendToSrc(ctx.attr * ctx.dstAttr), mergeMsg = _ + _, new TripletFields(false, true, true)).cache()

		val delta = g4.joinVertices(r3) { case (_, x, y) => math.abs(x - y)*math.abs(x - y) }.vertices.values.sum()
		
		val err = math.sqrt(delta)
		println( iter + ": " + err)
		
		if( err <= 1e-4 && (flag_1e4 == false)){
			outputVector(g4,"vec_blockpower_1e4.txt")
			flag_1e4 = true
		}
		
		if( err <= 1e-5 && (flag_1e5 == false)){
			outputVector(g4,"vec_blockpower_1e5.txt")
			flag_1e5 = true
		}
		
		if( err <= 1e-6 && (flag_1e6 == false)){
			outputVector(g4,"vec_blockpower_1e6.txt")
			flag_1e6 = true
		}
		
		if( err <= 1e-7 && (flag_1e7 == false)){
			outputVector(g4,"vec_blockpower_1e7.txt")
			flag_1e7 = true
		}
		
		if( err <= 1e-8 && (flag_1e8 == false)){
			outputVector(g4,"vec_blockpower_1e8.txt")
			flag_1e8 = true
		}
		
		if( err <= 1e-9 && (flag_1e9 == false)){
			outputVector(g4,"vec_blockpower_1e9.txt")
			flag_1e9 = true
		}
		
		err
  }
	

	
  def main(args: Array[String]) {
	val filename = "/user/hxxji001/" + args(0)
	val blocksize = args(4).toInt

    val conf = new SparkConf().setAppName("MatrixCompletion using randomized rrsvd algorithm")
    val sc = new SparkContext(conf)
	
	val g = loadMat(sc, filename,blocksize)
	val g2 = randomInit(g,blocksize)

	/*output(g2,"outputblock.txt")*/
	
	val v = randomized_rrsvd(sc, g2, blocksize, args(1).toInt, args(2).toInt, args(3).toDouble).sortBy(_._1)

	sc.stop()
  }
}
