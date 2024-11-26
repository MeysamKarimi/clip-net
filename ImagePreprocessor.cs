namespace CLIP;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;

public class ImagePreprocessor
{
    // Method to load, resize, and process images
    public static Tensor<float> LoadAndResizeImage(string imagePath, int width, int height)
    {
        try
        {
            // Load image as Rgba32 format (each pixel is a 32-bit RGBA value)
            using var image = Image.Load<Rgba32>(imagePath);

            // Resize the image to the target size (224x224 for CLIP)
            image.Mutate(x => x.Resize(width, height));

            // Initialize an array to hold the normalized pixel data (RGB channels)
            float[] imageData = new float[width * height * 3]; // 3 channels (RGB)

            int index = 0;

            // Accessing each pixel in the image after resize
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pixel = image[x, y]; // Access pixel at (x, y)

                    // Normalize pixel values to [0, 1]
                    imageData[index++] = pixel.R / 255f; // Red channel
                    imageData[index++] = pixel.G / 255f; // Green channel
                    imageData[index++] = pixel.B / 255f; // Blue channel
                }
            }

            // Convert the image data to a tensor format (channels, height, width)
            var imageTensor = new DenseTensor<float>(imageData, new[] { 1, 3, height, width });

            return imageTensor;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error processing image: {ex.Message}");
            return null;
        }
    }
}
