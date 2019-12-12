using System;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Numerics;
using System.Diagnostics;
using System.Threading.Tasks;

namespace SIMDBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            //var summary = BenchmarkRunner.Run<MinValueBenchmark>();
            //var summary = BenchmarkRunner.Run<SumBenchmark>();
            BenchmarkRunner.Run(typeof(Program).Assembly);

            Console.ReadKey();
        }
    }

    //public class MinValueBenchmark
    //{
    //    ushort[] source = Enumerable.Repeat(0, 2097152).Select(i => (ushort)new Random().Next(20, ushort.MaxValue - 1)).ToArray();

    //    [Benchmark]
    //    public unsafe ushort MinSIMD()
    //    {
    //        int volumen = source.Length;

    //        fixed (ushort* pSource = source)
    //        {
    //            do
    //            {
    //                volumen = (int)Math.Ceiling(volumen / 8m);

    //                for (var i = 0; i < volumen; i++)
    //                {
    //                    source[i] = Sse41.MinHorizontal(Sse2.LoadVector128(pSource + (i * 8))).GetElement(0);
    //                }
    //            }
    //            while (volumen > 8);

    //            return Sse41.MinHorizontal(Sse2.LoadVector128(pSource)).GetElement(0);
    //        }
    //    }

    //    [Benchmark]
    //    public unsafe ushort MinSIMDParalel()
    //    {
    //        int volumen = source.Length;

    //        fixed (ushort* pSource = source)
    //        {
    //            var ops = new UnsafeOps();
    //            ops.p = pSource;
    //            ops.source = source;

    //            do
    //            {
    //                volumen = (int)Math.Ceiling(volumen / 8m);

    //                Parallel.For(0, volumen - 1, ops.Lambda);
                    
    //            }
    //            while (volumen > 8);

    //            return Sse41.MinHorizontal(Sse2.LoadVector128(pSource)).GetElement(0);
    //        }
    //    }

    //    private unsafe class UnsafeOps
    //    {
    //        public ushort* p;
    //        public ushort[] source;

    //        public unsafe void Lambda(int shift)
    //        {
    //            source[shift] = Sse41.MinHorizontal(Sse2.LoadVector128(p + (shift * 8))).GetElement(0);
    //        }
    //    }

    //    [Benchmark]
    //    public ushort MinLINQ()
    //    {

    //        return source.Min();
    //    }

    //    [Benchmark]
    //    public ushort MinLoop()
    //    {

    //        ushort min = source[0];
    //        for(int i = 1; i < source.Length; i++)
    //        {
    //            if (source[i] < min)
    //                min = source[i];
    //        }
    //        return min;
    //    }
    //}

    public class SumBenchmark
    {
        internal float[] values;

        [Params(100_000)]
        public int N;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            values = Enumerable.Range(0, N).Select(_ => (float)rnd.NextDouble()).ToArray();
        }

        [Benchmark(Baseline = true)]
        public float SumNaive()
        {
            var values = this.values;
            float sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                sum += values[i];
            }
            return (float)sum;
        }

        [Benchmark]
        public float SumUsingLinq()
        {
            return values.Sum();
        }

        [Benchmark]
        public float SumUsingSystemNumerics()
        {
            var values = this.values;
            float sum = 0;

            // How may elements can we sum using vectors?
            int vectorizableLength = values.Length - values.Length % Vector<float>.Count;

            // Create vectors from the values and sum them into a temporary vector
            var tempVector = Vector<float>.Zero;
            var valuesSpan = values.AsSpan();
            for (int i = 0; i < vectorizableLength; i += Vector<float>.Count)
            {
                tempVector += new Vector<float>(valuesSpan.Slice(i, Vector<float>.Count));
            }

            // Sum the elements in the ttemporary vector
            for (int iVector = 0; iVector < Vector<float>.Count; iVector++)
            {
                sum += tempVector[iVector];
            }

            // Handle remaining elements
            for (int i = vectorizableLength; i < values.Length; i++)
            {
                sum += values[i];
            }

            return sum;
        }

        [Benchmark]
        public float SumUsingAvx()
        {
            var values = this.values;
            float sum = 0;

            if (Avx.IsSupported)
            {
                // How may elements can we sum using vectors?
                int vectorizableLength = values.Length - values.Length % Vector256<float>.Count;

                // Create vectors from the values and sum them into a temporary vector
                var tempVector = Vector256<float>.Zero;
                unsafe
                {
                    fixed (float* valuesPtr = values)
                    {
                        for (int i = 0; i < vectorizableLength; i += Vector256<float>.Count)
                        {
                            var valuesVector = Avx.LoadVector256(valuesPtr + i);
                            tempVector = Avx.Add(tempVector, valuesVector);
                        }
                    }
                }

                // Sum the elements in the ttemporary vector
                for (int iVector = 0; iVector < Vector256<float>.Count; iVector++)
                {
                    sum += tempVector.GetElement(iVector);
                }

                // Handle remaining elements
                for (int i = vectorizableLength; i < values.Length; i++)
                {
                    sum += values[i];
                }
            }
            else
            {
                // non-AVX capable machines
                for (int i = 0; i < values.Length; i++)
                {
                    sum += values[i];
                }
            }

            return sum;
        }

        [Benchmark]
        public float SumUsingAvxAlignedPipelined()
        {
            var values = this.values;
            float sum = 0;

            if (Avx.IsSupported)
            {
                unsafe
                {
                    fixed (float* valuesPtr = values)
                    {
                        // Determine how many elements we need to sum sequential to reach 256 bit alignment
                        const int ElementsPerByte = sizeof(float) / sizeof(byte);
                        var alignmentOffset = (long)(uint)(-(int)valuesPtr / ElementsPerByte) & (Vector256<float>.Count - 1);

                        // handle first values sequentially until we hit the 256bit alignment boundary
                        for (long i = 0; i < alignmentOffset; i++)
                        {
                            sum += *(valuesPtr + i);
                        }

                        var remainingLength = values.Length - alignmentOffset;

                        var vectorizableLength = values.Length - remainingLength % (long)Vector256<float>.Count;

                        // handle batches of 4 vectors for pipelining benefits
                        var pipelineVectorizableLength = values.Length - remainingLength % (4 * (long)Vector256<float>.Count);

                        var tempVector1 = Vector256<float>.Zero;
                        var tempVector2 = Vector256<float>.Zero;
                        var tempVector3 = Vector256<float>.Zero;
                        var tempVector4 = Vector256<float>.Zero;
                        for (long i = alignmentOffset; i < pipelineVectorizableLength; i += 4 * (long)Vector256<float>.Count)
                        {
                            var valuesVector1 = Avx.LoadAlignedVector256(valuesPtr + i);
                            var valuesVector2 = Avx.LoadAlignedVector256(valuesPtr + i + Vector256<float>.Count);
                            var valuesVector3 = Avx.LoadAlignedVector256(valuesPtr + i + 2 * Vector256<float>.Count);
                            var valuesVector4 = Avx.LoadAlignedVector256(valuesPtr + i + 3 * Vector256<float>.Count);
                            tempVector1 = Avx.Add(tempVector1, valuesVector1);
                            tempVector2 = Avx.Add(tempVector2, valuesVector2);
                            tempVector3 = Avx.Add(tempVector3, valuesVector3);
                            tempVector4 = Avx.Add(tempVector4, valuesVector4);
                        }

                        var tempVector = Avx.Add(Avx.Add(Avx.Add(tempVector1, tempVector2), tempVector3), tempVector4);

                        // handle remaining vectors
                        for (long i = pipelineVectorizableLength; i < vectorizableLength; i += Vector256<float>.Count)
                        {
                            var valuesVector = Avx.LoadAlignedVector256(valuesPtr + i);
                            tempVector = Avx.Add(tempVector, valuesVector);
                        }

                        // sum the temporary vector
                        for (int iVector = 0; iVector < Vector256<float>.Count; iVector++)
                        {
                            sum += tempVector.GetElement(iVector);
                        }

                        // handle remaining items
                        for (int i = (int)vectorizableLength; i < values.Length; i++)
                        {
                            sum += values[i];
                        }
                    }
                }
            }
            else
            {
                // non-AVX capable machines
                for (int i = 0; i < values.Length; i++)
                {
                    sum += values[i];
                }
            }

            return sum;
        }
    }   
}
