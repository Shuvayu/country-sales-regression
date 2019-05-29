using Microsoft.ML;
using SalesForecasting.Entities.Country;
using SalesForecasting.Extensions;
using System;
using System.IO;

namespace SalesForecasting.ModelTraining
{
    internal class CountrySalesPredictionModelPrep
    {
        /// <summary>
        /// Train and save model for predicting next month country unit sales
        /// </summary>
        /// <param name="dataPath">Input training file path</param>
        /// <param name="outputModelPath">Trained model path</param>
        /// <param name="mlContext"></param>
        public static void TrainAndSaveModel(MLContext mlContext, string dataPath, string outputModelPath = "country_sales_ml_model.zip")
        {
            if (File.Exists(outputModelPath))
            {
                File.Delete(outputModelPath);
            }

            CreateCountryModel(mlContext, dataPath, outputModelPath);
        }

        private static void CreateCountryModel(MLContext mlContext, string dataPath, string outputModelPath)
        {
            ConsoleExtensions.ConsoleWriteHeader("Training country forecasting model");

            var trainindDataView = mlContext.Data.LoadFromTextFile<CountryData>(path: dataPath, hasHeader: true, separatorChar: ',');

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie("Label", "Features", learningRate:0.2);

            var trainingPipeline = mlContext.Transforms.Concatenate(outputColumnName: "NumFeatures", nameof(CountryData.year),
                                                             nameof(CountryData.month), nameof(CountryData.max), nameof(CountryData.min),
                                                             nameof(CountryData.std), nameof(CountryData.count), nameof(CountryData.sales),
                                                             nameof(CountryData.med), nameof(CountryData.prev))
                                                        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CatFeatures", inputColumnName: nameof(CountryData.country)))
                                                        .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "NumFeatures", "CatFeatures"))
                                                        .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(CountryData.next)))
                                                        .Append(trainer);

            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");

            var crossValidationResults = mlContext.Regression.CrossValidate(data: trainindDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");

            ConsoleExtensions.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            // Create and Train the model
            var model = trainingPipeline.Fit(trainindDataView);

            //Save model
            mlContext.Model.Save(model, trainindDataView.Schema, outputModelPath);
        }

        /// <summary>
        /// Predict samples using saved model
        /// </summary>
        /// <param name="outputModelPath">Model file path</param>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static void TestPrediction(MLContext mlContext, string outputModelPath = "country_sales_ml_model.zip")
        {
            ConsoleExtensions.ConsoleWriteHeader("Testing Country Sales Forecast model");

            ITransformer trainedModel;
            using (var stream = File.OpenRead(outputModelPath))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<CountryData, CountrySalesPrediction>(trainedModel);

            Console.WriteLine("** Testing Country 1 **");

            // Build sample data
            var dataSample = new CountryData()
            {
                country = "United Kingdom",
                month = 10,
                year = 2017,
                med = 309.945F,
                max = 587.902F,
                min = 135.640F,
                std = 1063.932092F,
                prev = 856548.78F,
                count = 1724,
                sales = 873612.9F,
            };

            var prediction = predictionEngine.Predict(dataSample);
            Console.WriteLine($"Country: {dataSample.country}, month to predict: {dataSample.month + 1}, year: {dataSample.year} - Real value (US$): {Math.Pow(6.0084501F, 10)}, Predicted Forecast (US$): {Math.Pow(prediction.Score, 10)}");

            dataSample = new CountryData()
            {
                country = "United Kingdom",
                month = 11,
                year = 2017,
                med = 288.72F,
                max = 501.488F,
                min = 134.5360F,
                std = 707.5642F,
                prev = 873612.9F,
                count = 2387,
                sales = 1019647.67F,
            };
            prediction = predictionEngine.Predict(dataSample);
            Console.WriteLine($"Country: {dataSample.country}, month to predict: {dataSample.month + 1}, year: {dataSample.year} - Predicted Forecast (US$):  {Math.Pow(prediction.Score, 10)}");

            Console.WriteLine(" ");

            Console.WriteLine("** Testing Country 2 **");
            dataSample = new CountryData()
            {
                country = "United States",
                month = 10,
                year = 2017,
                med = 400.17F,
                max = 573.63F,
                min = 340.395F,
                std = 340.3959F,
                prev = 4264.94F,
                count = 10,
                sales = 5322.56F
            };
            prediction = predictionEngine.Predict(dataSample);
            Console.WriteLine($"Country: {dataSample.country}, month to predict: {dataSample.month + 1}, year: {dataSample.year} - Real value (US$): {Math.Pow(3.805769F, 10)}, Predicted Forecast (US$): {Math.Pow(prediction.Score, 10)}");

            dataSample = new CountryData()
            {
                country = "United States",
                month = 11,
                year = 2017,
                med = 317.9F,
                max = 1135.99F,
                min = 249.44F,
                std = 409.75528F,
                prev = 5322.56F,
                count = 11,
                sales = 6393.96F,
            };
            prediction = predictionEngine.Predict(dataSample);
            Console.WriteLine($"Country: {dataSample.country}, month to predict: {dataSample.month + 1}, year: {dataSample.year} - Predicted Forecast (US$):  {Math.Pow(prediction.Score, 10)}");
        }
    }
}

