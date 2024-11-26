namespace CLIP;

using Microsoft.ML.OnnxRuntime.Tensors;

public class ImageCaptionPair
{
    public Tensor<float> ImageTensor { get; set; }
    public List<Tensor<long>> TextTensors { get; set; }

    public ImageCaptionPair(Tensor<float> imageTensor, List<Tensor<long>> textTensors)
    {
        ImageTensor = imageTensor;
        TextTensors = textTensors;
    }
}

public class Dataset
{
    public List<ImageCaptionPair> ImageCaptionPairs { get; set; }

    public Dataset()
    {
        ImageCaptionPairs = new List<ImageCaptionPair>();
    }

    // Method to load images and captions into a dataset
    public void LoadDataset(string imagesFolder, string captionsFile)
    {
        var captions = CaptionReader.LoadCaptions(captionsFile);

        foreach (var imageName in captions.Keys)
        {
            var imagePath = Path.Combine(imagesFolder, imageName);  // Construct image path
            var imageTensor = ImagePreprocessor.LoadAndResizeImage(imagePath, 224, 224);  // R

            if (imageTensor != null)
            {
                var caption = captions[imageName];
                var captionTensors = new List<Tensor<long>>();

                // Tokenize each caption and convert to tensor
                foreach (var cap in caption)
                {
                    var tokenizedCaption = TextProcessor.TokenizeText(cap);
                    captionTensors.Add(tokenizedCaption);
                }

                var imageCaptionPair = new ImageCaptionPair(imageTensor, captionTensors);
                ImageCaptionPairs.Add(imageCaptionPair);
            }
        }
    }
}