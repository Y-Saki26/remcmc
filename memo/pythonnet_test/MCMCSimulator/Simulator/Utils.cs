using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Simulator
{
    public class Utils
    {
        // 一般のコンテナ操作etc
        public static double Variance<T>(IEnumerable<T> list) {
            var src_double = list.Select(x => Convert.ToDouble(x)).ToArray();
            double rms = src_double.Select(x => x * x).Average();
            double mean = src_double.Average();
            return rms - mean * mean;
        }

        public static double Stdev<T>(IEnumerable<T> list) {
            return Math.Sqrt(Variance(list));
        }

        /// <summary>
        /// Range like python
        /// </summary>
        /// <param name="start">start index (if end=null, it means end index)</param>
        /// <param name="end">end index</param>
        /// <param name="step">step index</param>
        /// <returns></returns>
        public static IEnumerable<int> ARange(int start, int? end = null, int step = 1) {
            if(end == null) {
                (start, end) = (0, start);
            }
            for(int i = start; i < end; i += step) {
                yield return i;
            }
        }

        /// <summary>
        /// Range like python
        /// </summary>
        /// <param name="start">start index (if end=null, it means end index)</param>
        /// <param name="end">end index</param>
        /// <param name="step">step index</param>
        /// <returns></returns>
        public static IEnumerable<decimal> ARange(decimal start, decimal? end = null, decimal step = 1) {
            if(end == null) {
                (start, end) = (0, start);
            }
            for(decimal i = start; i < end; i += step) {
                yield return i;
            }
        }


        public static bool Equals<T>(IEnumerable<T> list1, IEnumerable<T> list2) {
            return list1.Except(list2).Count() == 0 && list2.Except(list1).Count() == 0;
        }

        /// <summary>
        /// return values s.t. for(int i=start; i<end; i+=step)
        /// like `list[start: end: step]` in Python 
        /// </summary>
        public static IEnumerable<T> Slice<T>(IEnumerable<T> list, int start, int? end = null, int? skip = null) {
            var length = list.Count();
            if(end != null && end <= start || skip != null && (skip < 1 || skip > length)) {
                return new List<T>();
            }
            int _end = (end ?? 0) % length;
            if(_end == 0) _end += length;
            int _skip = skip ?? 1;
            return list.Where((value, index) => start <= index && index < _end && (index - start) % _skip == 0);
        }

        // 出力操作
        public static void PrintList<T>(IEnumerable<T> list, string sep = " ") {
            Console.WriteLine(string.Join(", ", list));
        }

        public static void PrintListList<T>(IEnumerable<IEnumerable<T>> list_array, string sep = " ") {
            foreach(var list in list_array)
                PrintList(list, sep);
        }

        // MCMCクラス中で使うもの
        public static float Energy(float j, float field, int interaction, int magnetization) => -j * interaction - field * magnetization;


        public static bool MetropolisTest(Random rnd, float oldLogLikelihood, float newLogLikelihood) {
            return newLogLikelihood >= oldLogLikelihood || (float)rnd.NextDouble() <= Math.Exp(newLogLikelihood - oldLogLikelihood);
        }

        public static bool ExchangeTest(Random rnd, float leftBeta, float rightBeta, float leftEnergy, float rightEnergy) {
            float deltaLogLilelihood = -(rightBeta - leftBeta) * (- rightEnergy + leftEnergy);
            return deltaLogLilelihood >= 0 || (float)rnd.NextDouble() <= Math.Exp(deltaLogLilelihood);
        }

        public static double SpecificHeat(IEnumerable<double> energy, double beta, int spin_num)
            => beta * beta * Variance(energy) / spin_num;

        // for REMCMC
        public static List<T>[] GetSeries<T>(List<T>[] series_n_k, List<int>[] chain_index_n_k) {
            int size_beta = series_n_k.Length, size_sample = series_n_k[0].Count();

            var return_series = new List<T>[size_beta];
            for(int k = 0; k < size_beta; k++) {
                return_series[k] = new List<T>();
            }
            for(int n = 0; n < size_sample; n++) {
                for(int k = 0; k < size_beta; k++) {
                    return_series[k].Add(series_n_k[chain_index_n_k[k][n]][n]);
                }
            }
            return return_series;
        }
    }
}
