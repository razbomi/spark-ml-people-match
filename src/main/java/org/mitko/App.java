package org.mitko;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import java.util.Arrays;
import java.util.List;

import static java.util.Arrays.asList;
import static org.apache.spark.sql.functions.*;

/**
 * Can read as sql
 * https://spark.apache.org/docs/latest/quick-start.html
 * https://spark.apache.org/docs/latest/rdd-programming-guide.html
 * https://spark.apache.org/examples.html
 * https://github.com/apache/spark/tree/master/examples/src/main/java/org/apache/spark/examples
 * https://github.com/amplab/drizzle-spark/blob/master/examples/src/main/java/org/apache/spark/examples/sql/JavaSparkSQLExample.java
 * https://github.com/ypriverol/spark-java8/blob/master/src/main/java/org/sps/learning/spark/basic/SparkWordCount.java
 * <p>
 * Record linking
 * https://towardsdatascience.com/record-linking-with-apache-sparks-mllib-graphx-d118c5f31f83
 * <p>
 * Text analysis
 * https://commons.apache.org/proper/commons-text/apidocs/org/apache/commons/text/similarity/package-summary.html
 */
public class App {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("Simple Application").master("local[4]").getOrCreate();

        // Machine Learning party
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.01);
//                .setElasticNetParam(0.8);

        LogisticRegressionModel model = lr.fit(trainingData(spark));
        Dataset<Row> predictions = model.transform(data(spark));
        result(predictions);

        spark.stop();
    }

    private static void result(Dataset<Row> predictions) {
        predictions.select(
                concat(col("l.first").as("l_first"), lit(" "), col("l.last").as("l_last")).as("left_name"),
                concat(col("r.first").as("r_first"), lit(" "), col("r.last").as("r_last")).as("right_name"),
                col("probability"),
                col("prediction")
        ).show(100);
    }

    private static Dataset<Row> data(SparkSession spark) {
        // Initial data, soundex may be nice
        Dataset<Row> jsonData = spark
                .read()
                .json("people.json")
                .withColumn("first_soundex", soundex(col("first")))
                .withColumn("last_soundex", soundex(col("last")))
                .cache();
        jsonData.show();

        // Crazy cartesian, join but needs a good condition to avoid cartesian poduct.
        Dataset<Row> left = jsonData.alias("l");
        Dataset<Row> right = jsonData.alias("r");
        Dataset<Row> join =
//        left.join(right, col("l.id").notEqual(col("r.id")))
                left.crossJoin(right)
//                .where(levenshtein(col("l.first"), col("r.first")).leq(2)).and(levenshtein(col("last1"), col("last")).leq(2)))
                        .withColumn("first_soundex_eq", col("l.first_soundex").equalTo(col("r.first_soundex")).cast(DataTypes.DoubleType))
                        .withColumn("last_soundex_eq", col("l.last_soundex").equalTo(col("r.last_soundex")).cast(DataTypes.DoubleType))
                        .withColumn("first_levenstein", levenshtein(col("l.first"), col("r.first")).cast(DataTypes.DoubleType))
                        .withColumn("last_levenstein", levenshtein(col("l.last"), col("r.last")).cast(DataTypes.DoubleType));
        join.show();

        // Features colum
        join = new VectorAssembler()
                .setInputCols(new String[]{"first_soundex_eq", "last_soundex_eq", "first_levenstein", "last_levenstein"})
                .setOutputCol("features")
                .transform(join);

        join.show(100);
        return join;
    }

    private static Dataset<Row> trainingData(SparkSession spark) {
        List<Row> dataTraining = Arrays.asList(
                RowFactory.create(1.0, Vectors.dense(1.0, 1.0, 0.0, 0.0)),
                RowFactory.create(1.0, Vectors.dense(1.0, 0.0, 1.0, 1.0)),
//                RowFactory.create(1.0, Vectors.dense(0.0, 0.0, 1.0, 1.0)),
                RowFactory.create(1.0, Vectors.dense(0.0, 0.0, 0.0, 1.0)),
                RowFactory.create(1.0, Vectors.dense(0.0, 0.0, 1.0, 0.0)),
                RowFactory.create(0.0, Vectors.dense(1.0, 0.0, 0.0, 10.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 1.0, 2.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 1.0, 10.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 5.0, 5.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 5.0, 6.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 5.0, 7.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 5.0, 10.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 5.0, 11.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 6.0, 4.0)),
                RowFactory.create(0.0, Vectors.dense(0.0, 0.0, 6.0, 5.0))
        );
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        return spark.createDataFrame(dataTraining, schema);
    }
}

