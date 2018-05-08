using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace ml.net
{
    class Program
    {

        public class IrisData
        {
            [Column("0")]
            public float SepalLength;

            [Column("1")]
            public float SepalWidth;

            [Column("2")]
            public float PetalLength;

            [Column("3")]
            public float PetalWidth;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            var pipeline = new LearningPipeline();
            string dataPath = "iris-data.txt";
            pipeline.Add(new TextLoader<IrisData>(dataPath, separator: ","));

            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<IrisData, IrisPrediction>();

            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"1st Predicted flower type is: {prediction.PredictedLabels}");

            var prediction2 = model.Predict(new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.2f,
                PetalLength = 1.5f,
                PetalWidth = 0.2f,
            });

            Console.WriteLine($"2nd Predicted flower type is: {prediction2.PredictedLabels}");
        }
    }
}
