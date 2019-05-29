using Microsoft.ML;
using SalesForecasting.Extensions;
using SalesForecasting.ModelTraining;
using System;
using System.IO;

namespace SalesForecasting
{
    internal class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"Data";
        private static readonly string CountryDataRealtivePath = $"{BaseDatasetsRelativePath}\\country.stats.csv";
        private static readonly string ProductDataRealtivePath = $"{BaseDatasetsRelativePath}\\product.stats.csv";

        private static readonly string CountryDataPath = GetAbsolutePath(CountryDataRealtivePath);
        private static readonly string ProductDataPath = GetAbsolutePath(ProductDataRealtivePath);

        private static void Main()
        {

            try
            {
                MLContext mlContext = new MLContext(seed: 2);  //Seed set to any number so you have a deterministic environment

                CountrySalesPredictionModelPrep.TrainAndSaveModel(mlContext, CountryDataPath);
                CountrySalesPredictionModelPrep.TestPrediction(mlContext);
            }
            catch (Exception ex)
            {
                ConsoleExtensions.ConsoleWriteException(ex.Message);
            }
            ConsoleExtensions.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativeDatasetPath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);

            return fullPath;
        }
    }
}
