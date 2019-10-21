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
            var summary = BenchmarkRunner.Run<MinValueBenchmark>();
            //var summary = BenchmarkRunner.Run<SumBenchmark>();
            Console.ReadKey();
        }
    }

    public class MinValueBenchmark
    {
        ushort[] source = Enumerable.Repeat(0, 2097152).Select(i => (ushort)new Random().Next(20, ushort.MaxValue - 1)).ToArray();

        [Benchmark]
        public unsafe ushort MinSIMD()
        {
            int volumen = source.Length;

            fixed (ushort* pSource = source)
            {
                do
                {
                    volumen = (int)Math.Ceiling(volumen / 8m);

                    for (var i = 0; i < volumen; i++)
                    {
                        source[i] = Sse41.MinHorizontal(Sse2.LoadVector128(pSource + (i * 8))).GetElement(0);
                    }
                }
                while (volumen > 8);

                return Sse41.MinHorizontal(Sse2.LoadVector128(pSource)).GetElement(0);
            }
        }

        [Benchmark]
        public unsafe ushort MinSIMDParalel()
        {
            int volumen = source.Length;

            fixed (ushort* pSource = source)
            {
                var ops = new UnsafeOps();
                ops.p = pSource;
                ops.source = source;

                do
                {
                    volumen = (int)Math.Ceiling(volumen / 8m);

                    Parallel.For(0, volumen - 1, ops.Lambda);
                    
                }
                while (volumen > 8);

                return Sse41.MinHorizontal(Sse2.LoadVector128(pSource)).GetElement(0);
            }
        }

        private unsafe class UnsafeOps
        {
            public ushort* p;
            public ushort[] source;

            public unsafe void Lambda(int shift)
            {
                source[shift] = Sse41.MinHorizontal(Sse2.LoadVector128(p + (shift * 8))).GetElement(0);
            }
        }

        [Benchmark]
        public ushort MinLINQ()
        {

            return source.Min();
        }

        [Benchmark]
        public ushort MinLoop()
        {

            ushort min = source[0];
            for(int i = 1; i < source.Length; i++)
            {
                if (source[i] < min)
                    min = source[i];
            }
            return min;
        }
    }

    public class SumBenchmark
    {
        int[] array = Enumerable.Repeat(0, 2097152).Select(i => new Random().Next(100)).ToArray();

        [Benchmark]
        public int Sum()
        {
            var source = new ReadOnlySpan<int>(array);
            int result = 0;

            for (int i = 0; i < source.Length; i++)
            {
                result += source[i];
            }

            return result;
        }

        [Benchmark]
        public unsafe int SumUnrolled()
        {
            var source = new ReadOnlySpan<int>(array);
            int result = 0;

            int i = 0;
            int lastBlockIndex = source.Length - (source.Length % 4);

            // Pin source so we can elide the bounds checks
            fixed (int* pSource = source)
            {
                while (i < lastBlockIndex)
                {
                    result += pSource[i + 0];
                    result += pSource[i + 1];
                    result += pSource[i + 2];
                    result += pSource[i + 3];

                    i += 4;
                }

                while (i < source.Length)
                {
                    result += pSource[i];
                    i += 1;
                }
            }

            return result;
        }

        [Benchmark]
        public int SumVectorT()
        {
            var source = new ReadOnlySpan<int>(array);
            int result = 0;

            Vector<int> vresult = Vector<int>.Zero;

            int i = 0;
            int lastBlockIndex = source.Length - (source.Length % Vector<int>.Count);

            while (i < lastBlockIndex)
            {
                vresult += new Vector<int>(source.Slice(i));
                i += Vector<int>.Count;
            }

            for (int n = 0; n < Vector<int>.Count; n++)
            {
                result += vresult[n];
            }

            while (i < source.Length)
            {
                result += source[i];
                i += 1;
            }

            return result;
        }

        [Benchmark]
        public int SumVectorized()
        {
            var source = new ReadOnlySpan<int>(array);
            if (Avx2.IsSupported)
            {
                return SumVectorizedAvx2(source);
            }
            if (Sse2.IsSupported)
            {
                return SumVectorizedSse2(source);
            }
            else
            {
                return SumVectorT();
            }
        }

        private unsafe int SumVectorizedSse2(ReadOnlySpan<int> source)
        {
            int result;

            fixed (int* pSource = source)
            {
                Vector128<int> vresult = Vector128<int>.Zero;

                int i = 0;
                int lastBlockIndex = source.Length - (source.Length % 4);

                while (i < lastBlockIndex)
                {
                    vresult = Sse2.Add(vresult, Sse2.LoadVector128(pSource + i));
                    i += 4;
                }

                if (Ssse3.IsSupported)
                {
                    vresult = Ssse3.HorizontalAdd(vresult, vresult);
                    vresult = Ssse3.HorizontalAdd(vresult, vresult);
                }
                else
                {
                    vresult = Sse2.Add(vresult, Sse2.Shuffle(vresult, 0x4E));
                    vresult = Sse2.Add(vresult, Sse2.Shuffle(vresult, 0xB1));
                }
                result = vresult.ToScalar();

                while (i < source.Length)
                {
                    result += pSource[i];
                    i += 1;
                }
            }

            return result;
        }

        private unsafe int SumVectorizedAvx2(ReadOnlySpan<int> source)
        {
            int result;

            fixed (int* pSource = source)
            {
                Vector256<int> vresult = Vector256<int>.Zero;

                int i = 0;
                int lastBlockIndex = source.Length - (source.Length % 8);

                while (i < lastBlockIndex)
                {
                    vresult = Avx2.Add(vresult, Avx.LoadAlignedVector256(pSource + i));
                    i += 8;
                }

                vresult = Avx2.HorizontalAdd(vresult, vresult);
                vresult = Avx2.HorizontalAdd(vresult, vresult);
                result = vresult.ToScalar();

                while (i < source.Length)
                {
                    result += pSource[i];
                    i += 1;
                }
            }

            return result;
        }
    }   
}
