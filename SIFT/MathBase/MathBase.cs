using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV.XFeatures2D;


namespace MathBase
{
 
    public enum ColorFormat
    {
        RGB,
        GrayScale
    }
    public class SIFTSolver
    {
        public PointF Center { get; set; } = new PointF(0, 0);
        public float Sigma { get; set; } = 1;

        public ColorFormat ColorFormat { get; set; } = ColorFormat.RGB;



        public SIFTSolver(float xi, float yi, float sigma)
        {
            Center = new PointF(xi, yi);
            Sigma = sigma;
        }

        public SIFTSolver(float sigma)
        {
            Sigma = sigma;
        }

        public float GaussianSolve(float x, float y, float xi, float yi)
        {
            return Convert.ToSingle(1 / (2 * Math.PI * Math.Pow(Sigma, 2)) * Math.Pow(Math.E, 0 - ((Math.Pow(x - xi, 2) + Math.Pow(y - yi, 2)) / (2 * Math.Pow(Sigma, 2)))));
        }

        public float GaussianSolve(float x, float y)
        {
            return Convert.ToSingle((1 / (2 * Math.PI * Math.Pow(Sigma, 2))) * (Math.Pow(Math.E, 0 - ((Math.Pow(x - Center.X, 2) + Math.Pow(y - Center.Y, 2)) / (2 * Math.Pow(Sigma, 2))))));
        }

        public int[,] GetGaussianTable()
        {
            int range = (int)(3 * Sigma + 1);
            int[,] gaussianMap = new int[2 * range + 1, 2 * range + 1];

            for (int i = 0; i < 2 * range + 1; i++)
            {
                for (int j = 0; j < 2 * range + 1; j++)
                {
                    gaussianMap[i, j] = (int)(GaussianSolve(i, j, range, range) * 100000);
                }
            }
            return gaussianMap;
        }

        public int[,] GetGaussianTable(float sigma)
        {
            int range = (int)(3 * sigma + 1);
            int[,] gaussianMap = new int[2 * range + 1, 2 * range + 1];

            for (int i = 0; i < 2 * range + 1; i++)
            {
                for (int j = 0; j < 2 * range + 1; j++)
                {
                    gaussianMap[i, j] = (int)(GaussianSolve(i, j, range, range) * 100000);
                }
            }
            return gaussianMap;
        }

        public Bitmap[,] GetGaussianPyramid(Bitmap originBit,int octaves,int layers)
        {
            Bitmap[,] gaussianPyramid = new Bitmap[octaves, layers];
            for (int i = 0; i < octaves; i++)
            {
                float sigmaInit = 1.0f;
                for (int j = 0; j < layers; j++)
                {
                    originBit = Conv(originBit, (j + 1) * Sigma / sigmaInit);
                    gaussianPyramid[i, j] = new Bitmap(Bitmap.FromHbitmap(originBit.GetHbitmap()));
                    sigmaInit = (j + 1) * Sigma / sigmaInit;
                }
                originBit = new Bitmap(Bitmap.FromHbitmap(gaussianPyramid[i, layers - 2].GetHbitmap()));
                originBit = UnityTools.ImageScale(originBit, 0.5f);
            }

            return gaussianPyramid;
        }

        public Bitmap Conv(Bitmap originBitmap, float sigma)
        {
            int range = (int)(3 * sigma + 1);
            int[,] gaussianMap = new int[2 * range + 1, 2 * range + 1];

            for (int i = 0; i < 2 * range + 1; i++)
            {
                for (int j = 0; j < 2 * range + 1; j++)
                {
                    gaussianMap[i, j] = (int)(GaussianSolve(i, j, range, range) * 100000);
                }
            }
            int[,,] rgbMapInput = new int[originBitmap.Width, originBitmap.Height, 3];
            int[,,] rgbMapOutput = new int[originBitmap.Width, originBitmap.Height, 3];

            int maxIn = 0;
            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    if (ColorFormat == ColorFormat.RGB)
                    {
                        rgbMapInput[j, i, 0] = originBitmap.GetPixel(j, i).R;
                        rgbMapInput[j, i, 1] = originBitmap.GetPixel(j, i).G;
                        rgbMapInput[j, i, 2] = originBitmap.GetPixel(j, i).B;
                        maxIn = Math.Max(rgbMapInput[j, i, 0], maxIn);
                        maxIn = Math.Max(rgbMapInput[j, i, 1], maxIn);
                        maxIn = Math.Max(rgbMapInput[j, i, 2], maxIn);
                    }
                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        rgbMapInput[j, i, 0] = originBitmap.GetPixel(j, i).R;
                        maxIn = Math.Max(rgbMapInput[j, i, 0], maxIn);
                    }               
                }
            }

            Bitmap convedBitmap = new Bitmap(originBitmap.Width, originBitmap.Height);
            int maxOut = 0;
            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    for (int ii = 0; ii < 2 * range + 1; ii++)
                    {
                        for (int jj = 0; jj < 2 * range + 1; jj++)
                        {
                            int x = j + jj - range, y = i + ii - range;
                            if (x < 0)
                            {
                                x = -x;
                            }

                            if (y < 0)
                            {
                                y = -y;
                            }

                            if (x >= originBitmap.Width)
                            {
                                x = 2 * originBitmap.Width - x - 1;
                            }
                            if (y >= originBitmap.Height)
                            {
                                y = 2 * originBitmap.Height - y - 1;
                            }

                            if (ColorFormat == ColorFormat.RGB)
                            {
                                rgbMapOutput[j, i, 0] += gaussianMap[jj, ii] * rgbMapInput[x, y, 0];
                                rgbMapOutput[j, i, 1] += gaussianMap[jj, ii] * rgbMapInput[x, y, 1];
                                rgbMapOutput[j, i, 2] += gaussianMap[jj, ii] * rgbMapInput[x, y, 2];
                            }

                            if (ColorFormat == ColorFormat.GrayScale)
                            {
                                rgbMapOutput[j, i, 0] += gaussianMap[jj, ii] * rgbMapInput[x, y, 0];
                            }               
                        }
                    }

                    if (ColorFormat == ColorFormat.RGB)
                    {
                        maxOut = Math.Max(rgbMapOutput[j, i, 0], maxOut);
                        maxOut = Math.Max(rgbMapOutput[j, i, 1], maxOut);
                        maxOut = Math.Max(rgbMapOutput[j, i, 2], maxOut);
                    }

                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        maxOut = Math.Max(rgbMapOutput[j, i, 0], maxOut);
                    }                
                }
            }

            int[,,] rgbMapOutputs = UnityTools.TupleScale(rgbMapOutput, maxOut, maxIn);

            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    Color color;
                    if (ColorFormat == ColorFormat.RGB)
                    {
                        color = Color.FromArgb(Convert.ToInt16(originBitmap.GetPixel(j, i).A), rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 1], rgbMapOutputs[j, i, 2]);
                        convedBitmap.SetPixel(j, i, color);
                    }

                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        color = Color.FromArgb(Convert.ToInt16(originBitmap.GetPixel(j, i).A), rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 0]);
                        convedBitmap.SetPixel(j, i, color);
                    }
                    
                }
            }
            return convedBitmap;
        }

        public Bitmap Conv(Bitmap originBitmap)
        {
            int range = (int)(3 * Sigma + 1);
            int[,] gaussianMap = new int[2 * range + 1, 2 * range + 1];

            for (int i = 0; i < 2 * range + 1; i++)
            {
                for (int j = 0; j < 2 * range + 1; j++)
                {
                    gaussianMap[i, j] = (int)(GaussianSolve(i, j, range, range) * 100000);
                }
            }
            int[,,] rgbMapInput = new int[originBitmap.Width, originBitmap.Height, 3];
            int[,,] rgbMapOutput = new int[originBitmap.Width, originBitmap.Height, 3];

            int maxIn = 0;
            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    if (ColorFormat == ColorFormat.RGB)
                    {
                        rgbMapInput[j, i, 0] = originBitmap.GetPixel(j, i).R;
                        rgbMapInput[j, i, 1] = originBitmap.GetPixel(j, i).G;
                        rgbMapInput[j, i, 2] = originBitmap.GetPixel(j, i).B;
                        maxIn = Math.Max(rgbMapInput[j, i, 0], maxIn);
                        maxIn = Math.Max(rgbMapInput[j, i, 1], maxIn);
                        maxIn = Math.Max(rgbMapInput[j, i, 2], maxIn);
                    }
                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        rgbMapInput[j, i, 0] = originBitmap.GetPixel(j, i).R;
                        maxIn = Math.Max(rgbMapInput[j, i, 0], maxIn);
                    }
                }
            }

            Bitmap convedBitmap = new Bitmap(originBitmap.Width, originBitmap.Height);
            int maxOut = 0;
            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    for (int ii = 0; ii < 2 * range + 1; ii++)
                    {
                        for (int jj = 0; jj < 2 * range + 1; jj++)
                        {
                            int x = j + jj - range, y = i + ii - range;
                            if (x < 0)
                            {
                                x = -x;
                            }

                            if (y < 0)
                            {
                                y = -y;
                            }

                            if (x >= originBitmap.Width)
                            {
                                x = 2 * originBitmap.Width - x - 1;
                            }
                            if (y >= originBitmap.Height)
                            {
                                y = 2 * originBitmap.Height - y - 1;
                            }

                            if (ColorFormat == ColorFormat.RGB)
                            {
                                rgbMapOutput[j, i, 0] += gaussianMap[jj, ii] * rgbMapInput[x, y, 0];
                                rgbMapOutput[j, i, 1] += gaussianMap[jj, ii] * rgbMapInput[x, y, 1];
                                rgbMapOutput[j, i, 2] += gaussianMap[jj, ii] * rgbMapInput[x, y, 2];
                            }

                            if (ColorFormat == ColorFormat.GrayScale)
                            {
                                rgbMapOutput[j, i, 0] += gaussianMap[jj, ii] * rgbMapInput[x, y, 0];
                            }
                        }
                    }

                    if (ColorFormat == ColorFormat.RGB)
                    {
                        maxOut = Math.Max(rgbMapOutput[j, i, 0], maxOut);
                        maxOut = Math.Max(rgbMapOutput[j, i, 1], maxOut);
                        maxOut = Math.Max(rgbMapOutput[j, i, 2], maxOut);
                    }

                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        maxOut = Math.Max(rgbMapOutput[j, i, 0], maxOut);
                    }
                }
            }

            int[,,] rgbMapOutputs = UnityTools.TupleScale(rgbMapOutput, maxOut, maxIn);

            for (int i = 0; i < originBitmap.Height; i++)
            {
                for (int j = 0; j < originBitmap.Width; j++)
                {
                    Color color;
                    if (ColorFormat == ColorFormat.RGB)
                    {
                        color = Color.FromArgb(Convert.ToInt16(originBitmap.GetPixel(j, i).A), rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 1], rgbMapOutputs[j, i, 2]);
                        convedBitmap.SetPixel(j, i, color);
                    }

                    if (ColorFormat == ColorFormat.GrayScale)
                    {
                        color = Color.FromArgb(Convert.ToInt16(originBitmap.GetPixel(j, i).A), rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 0], rgbMapOutputs[j, i, 0]);
                        convedBitmap.SetPixel(j, i, color);
                    }

                }
            }
            return convedBitmap;
        }

        public Bitmap[,] GetDoGs(Bitmap[,] gaussainPyramid)
        {
            Bitmap[,] dogs = new Bitmap[gaussainPyramid.GetLength(0), gaussainPyramid.GetLength(1) - 1];
            for (int i = 0; i < dogs.GetLength(0); i++)
            {
                for (int j = 0; j < dogs.GetLength(1); j++)
                {
                    dogs[i, j] = UnityTools.Diff(gaussainPyramid[i, j], gaussainPyramid[i, j + 1]);
                }
            }
            return dogs;
        }


        
    }

    public static class UnityTools
    {
        public static Bitmap Diff(Bitmap bitmap1,Bitmap bitmap2)
        {
            Bitmap bitOut = new Bitmap(bitmap1.Width, bitmap1.Height);
            if (bitmap1.Size == bitmap2.Size)
            {

                int[,,] colorMap = new int[bitmap1.Width, bitmap1.Height, 3];
                int max = -255, min = 255;
                
                for (int i = 0; i < bitmap1.Height; i++)
                {
                    for (int j = 0; j < bitmap1.Width; j++)
                    {
                        colorMap[j, i, 0] = bitmap1.GetPixel(j, i).R - bitmap2.GetPixel(j, i).R;
                        colorMap[j, i, 1] = bitmap1.GetPixel(j, i).G - bitmap2.GetPixel(j, i).G;
                        colorMap[j, i, 2] = bitmap1.GetPixel(j, i).B - bitmap2.GetPixel(j, i).B;

                        min = Math.Min(min, colorMap[j, i, 0]);
                        min = Math.Min(min, colorMap[j, i, 1]);
                        min = Math.Min(min, colorMap[j, i, 2]);

                        max = Math.Max(max, colorMap[j, i, 0]);
                        max = Math.Max(max, colorMap[j, i, 1]);
                        max = Math.Max(max, colorMap[j, i, 2]); 
                    }
                }
                int[,,] colorMapOut = TupleScale(colorMap, 0, 255, min, max);


                for (int i = 0; i < colorMapOut.GetLength(1); i++)
                {
                    for (int j = 0; j < colorMapOut.GetLength(0); j++)
                    {
                        Color color = Color.FromArgb(255, colorMapOut[j, i, 0], colorMapOut[j, i, 1], colorMapOut[j, i, 2]);
                        bitOut.SetPixel(j, i, color);
                    }
                }

            }
            return bitOut;      
        }

        public static Bitmap ImageScale(Bitmap bitmap, float scaleFactor)
        {
            Bitmap bmp = new Bitmap((int)(bitmap.Width * scaleFactor), (int)(bitmap.Height * scaleFactor));
            Graphics g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            g.ScaleTransform(scaleFactor, scaleFactor);
            g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
            return bmp;
        }
        public static int[,,] TupleScale(int[,,] unScaled, int maxValueOut, int maxValueIn)
        {
            int[,,] scaled = new int[unScaled.GetLength(0), unScaled.GetLength(1), 3];

            for (int i = 0; i < unScaled.GetLength(1); i++)
            {
                for (int j = 0; j < unScaled.GetLength(0); j++)
                {

                    scaled[j, i, 0] = (int)((float)unScaled[j, i, 0] / maxValueOut * maxValueIn);
                    scaled[j, i, 1] = (int)((float)unScaled[j, i, 1] / maxValueOut * maxValueIn);
                    scaled[j, i, 2] = (int)((float)unScaled[j, i, 2] / maxValueOut * maxValueIn);

                }
            }

            return scaled;
        }
        public static int[,,] TupleScale(int[,,] unScaled, int rangeMin, int rangeMax, int inputMin, int inputMax)
        {
            int[,,] scaled = new int[unScaled.GetLength(0), unScaled.GetLength(1), 3];

            for (int i = 0; i < unScaled.GetLength(1); i++)
            {
                for (int j = 0; j < unScaled.GetLength(0); j++)
                {

                    scaled[j, i, 0] = (int)(((float)unScaled[j, i, 0] - inputMin) * (rangeMax - rangeMin) / (inputMax - inputMin) + rangeMin);
                    scaled[j, i, 1] = (int)(((float)unScaled[j, i, 1] - inputMin) * (rangeMax - rangeMin) / (inputMax - inputMin) + rangeMin);
                    scaled[j, i, 2] = (int)(((float)unScaled[j, i, 2] - inputMin) * (rangeMax - rangeMin) / (inputMax - inputMin) + rangeMin);

                }
            }


            return scaled;
        }

        public static Bitmap ConvertToGray(Bitmap rgbImage)
        {
            Bitmap grayImage = new Bitmap(rgbImage.Width, rgbImage.Height);
            for (int i = 0; i < rgbImage.Width; i++)
            {
                for (int j = 0; j < rgbImage.Height; j++)
                {
                    short gray = (short)((rgbImage.GetPixel(i, j).R * 19595 + rgbImage.GetPixel(i, j).G * 38469 + rgbImage.GetPixel(i, j).B * 7472) >> 16);
                    Color grayColor = Color.FromArgb(gray, gray, gray);
                    grayImage.SetPixel(i, j, grayColor);
                }
            }

            return grayImage;
            
        }
    }



}
