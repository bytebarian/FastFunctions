using System;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Numerics;
using System.Diagnostics;

namespace SIMDBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            //var summary = BenchmarkRunner.Run<SIMDMinValueBenchmark>();
            //var summary = BenchmarkRunner.Run<SIMDAddArrays>();
            var test = new SIMDAddArrays();
            test.SimdAdd();
            test.Add();

            Console.ReadKey();
        }
    }

    public class SIMDMinValueBenchmark
    {
        ushort us0 = 24; ushort us1 = 56; ushort us2 = 798; ushort us3 = 12567; ushort us4 = 8; ushort us5 = 1887; ushort us6 = 1; ushort us7 = 7;

        [Benchmark]
        public ushort MinSIMD()
        {

            var values = Vector128.Create(us0, us1, us2, us3, us4, us5, us6, us7);
            return Sse41.MinHorizontal(values).GetElement(0);
        }

        [Benchmark]
        public ushort MinLINQ()
        {

            return new ushort[8] { us0, us1, us2, us3, us4, us5, us6, us7 }.Min();
        }

        [Benchmark]
        public ushort MinLoop()
        {

            ushort min = us0;
            foreach (var value in new ushort[7] { us1, us2, us3, us4, us5, us6, us7 })
            {
                if (value < min)
                    min = value;
            }
            return min;
        }
    }

    public class SIMDAddArrays
    {
        static int N = 10;
        float[] a, b;

        public SIMDAddArrays()
        {
            var random = new Random();
            var a = Enumerable.Repeat(0, N).Select(i => random.Next(2 * N)).Select(x => (float)x).ToArray();
            var b = Enumerable.Repeat(0, N).Select(i => random.Next(2 * N)).Select(x => (float)x).ToArray();
        }

        [Benchmark]
        public float[] SimdAdd()
        {
            return SimdAddInternal(a, b, N);
        }

        [Benchmark]
        public float[] Add()
        {
            return AddInternal(a, b, N);
        }

        private unsafe float[] SimdAddInternal(float[] a, float[] b, int n)
        {
            float[] result = new float[n];
            fixed (float* ptr_a = a, ptr_b = b, ptr_res = result)
            {
                for (int i = 0; i < n; i += Vector256<float>.Count)
                {
                    Vector256<float> v1 = Avx.LoadVector256(ptr_a + i);
                    Vector256<float> v2 = Avx.LoadVector256(ptr_b + i);
                    Vector256<float> res = Avx.Add(v1, v2);
                    Avx.Store(ptr_res + i, res);
                }
            }
            return result;
        }

        private float[] AddInternal(float[] a, float[] b, int n)
        {
            float[] result = new float[n];

            for(int i = 0; i < n; i++)
            {
                result[i] = a[i] + b[i];
            }

            return result;
        }
    }
}
