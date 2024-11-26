using CLIP;

Console.WriteLine("Hello, .NET lovers who like to learn AI!");

string imagesFolder = @"\Models\Flickr8kDataset\Images";
string captionsFile = @"\Models\Flickr8kDataset\captions.txt";

var dataset = new Dataset();
dataset.LoadDataset(imagesFolder, captionsFile);

// STEP1: the dataset is ready to use
Console.WriteLine($"Loaded {dataset.ImageCaptionPairs.Count} image-caption pairs.");


// STEP2:  Load CLIP model
string modelPath = @"\Models\clip-image-vit-32-float32.onnx";
var clipModel = new CLIPModel(modelPath);

// STEP3: Compute similarities for the entire dataset
SimilarityCalculator.ComputeSimilarities(dataset, clipModel);
