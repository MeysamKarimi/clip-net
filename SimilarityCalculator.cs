namespace CLIP;

using System;
using Microsoft.ML.OnnxRuntime.Tensors;

public class SimilarityCalculator
{
    public static void ComputeSimilarities(Dataset dataset, CLIPModel clipModel)
    {
        foreach (var imageCaptionPair in dataset.ImageCaptionPairs)
        {
            // Get image embedding
            var imageEmbedding = clipModel.GetImageEmbedding(imageCaptionPair.ImageTensor);

            foreach (var captionTensor in imageCaptionPair.TextTensors)
            {
                // Get caption embedding
                var captionEmbedding = clipModel.GetTextEmbedding(captionTensor);

                // Compute cosine similarity
                float similarity = CosineSimilarity(imageEmbedding, captionEmbedding);
                Console.WriteLine($"Similarity between image and caption: {similarity}");
            }
        }
    }

    public static float CosineSimilarity(Tensor<float> tensor1, Tensor<float> tensor2)
    {
        // Flatten tensors to 1D arrays
        var vec1 = tensor1.ToArray();
        var vec2 = tensor2.ToArray();

        // Compute the dot product
        float dotProduct = vec1.Zip(vec2, (a, b) => a * b).Sum();

        // Compute the magnitudes of the tensors
        float magnitude1 = (float)Math.Sqrt(vec1.Sum(x => x * x));
        float magnitude2 = (float)Math.Sqrt(vec2.Sum(x => x * x));

        // Return the cosine similarity
        return dotProduct / (magnitude1 * magnitude2);
    }
}
