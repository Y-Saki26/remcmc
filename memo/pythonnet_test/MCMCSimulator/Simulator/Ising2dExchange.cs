using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using static Simulator.Utils;

namespace Simulator
{
    public class Ising2dExchange
    {
        public readonly int SizeWidth, SizeHeight, SizeBeta, ExchangeRate;
        public readonly float J, Field;
        public readonly float[] Beta_k;
        public int SampleCount;
        public List<int>[] BetaIndex_n_k, ChainIndex_n_k;

        private readonly int[] dx = { 0, 0, -1, 1 }, dy = { -1, 1, 0, 0 };
        private List<int>[] interaction_n_k, magnetization_n_k;
        private List<float>[] energy_n_k;
        private int[,,] spin_h_w_k;
        private int[] oldInteraction_k, oldMagnetization_k;
        private float[] oldEnergy_k;
        private Random rnd;
        public Ising2dExchange(
            int SizeWidth, int SizeHeight, float[] Beta_k, int ExchangeStep,
            float J = 1.0f, float Field = 0.0f) {
            rnd = new Random();
            this.SizeWidth = SizeWidth;
            this.SizeHeight = SizeHeight;
            this.Beta_k = Beta_k;
            SizeBeta = Beta_k.Length;
            this.ExchangeRate = ExchangeStep;
            this.J = J;
            this.Field = Field;

            spin_h_w_k = new int[SizeBeta, SizeWidth, SizeHeight];
            for(int k = 0; k < SizeBeta; k++) {
                for(int w = 0; w < SizeWidth; w++) {
                    for(int h = 0; h < SizeHeight; h++) {
                        spin_h_w_k[k, w, h] = rnd.Next(2) * 2 - 1;
                    }
                }
            }

            interaction_n_k = new List<int>[SizeBeta];
            magnetization_n_k = new List<int>[SizeBeta];
            energy_n_k = new List<float>[SizeBeta];
            BetaIndex_n_k = new List<int>[SizeBeta];
            ChainIndex_n_k = new List<int>[SizeBeta];
            oldInteraction_k = new int[SizeBeta];
            oldMagnetization_k = new int[SizeBeta];
            oldEnergy_k = new float[SizeBeta];
            for(int k = 0; k < SizeBeta; k++) {
                interaction_n_k[k] = new List<int>();
                interaction_n_k[k].Add(LastInteraction(k));
                oldInteraction_k[k] = interaction_n_k[k].Last();
                magnetization_n_k[k] = new List<int>();
                magnetization_n_k[k].Add(LastMagnetization(k));
                oldMagnetization_k[k] = magnetization_n_k[k].Last();
                energy_n_k[k] = new List<float>();
                energy_n_k[k].Add(LastEnergy(k));
                oldEnergy_k[k] = energy_n_k[k].Last();
                BetaIndex_n_k[k] = new List<int>();
                BetaIndex_n_k[k].Add(k);
                ChainIndex_n_k[k] = new List<int>();
                ChainIndex_n_k[k].Add(k);
            }
            SampleCount = 1;

#if DEBUG
            Verbose();
            for(int k = 0; k < SizeBeta; k++) {
                Debug.Assert(Interaction_n_k[k].Last() == LastInteraction(k), $"{SampleCount}, {k}: interaction not match, {Interaction_n_k[k].Last()}, {LastInteraction(k)}");
                Debug.Assert(Magnetization_n_k[k].Last() == LastMagnetization(k), $"{SampleCount}, {k}: magnet not match, {Magnetization_n_k[k].Last()}, {LastMagnetization(k)}");
                Debug.Assert(Energy_n_k[k].Last() == LastEnergy(k), $"{SampleCount}, {k}: energy not mathch,, {Energy_n_k[k].Last()}, {LastEnergy(k)}");
            }
            Console.WriteLine($"{SampleCount}: OK.");
#endif
        }

        public int LastInteraction(int k) {
            int sum = 0;
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    for(int kk = 0; kk < 4; kk++) {
                        int ww = w + dx[kk], hh = h + dy[kk];
                        if(0 <= ww && ww < SizeWidth && 0 <= hh && hh < SizeHeight) {
                            sum += spin_h_w_k[k, w, h] * spin_h_w_k[k, ww, hh];
                        }
                    }

                }
            }
            return sum / 2;
        }

        public int LastMagnetization(int k) {
            int sum = 0;
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    sum += spin_h_w_k[k, w, h];
                }
            }
            return sum;
        }

        public float LastEnergy(int k) => Energy(J, Field, oldInteraction_k[k], oldMagnetization_k[k]);

        public void Sampling(int sample_size, bool verbose = false, int verbose_frequency = 100) {
            for(int n = 0; n < sample_size; n++) {
                if(SampleCount % ExchangeRate != 0) {
                    SamplingLocal();
                } else {
                    SamplingExchange((SampleCount / ExchangeRate) % 2 == 0);
                }
                SampleCount++;

                if(verbose && SampleCount % verbose_frequency == 0) {
                    Verbose();
#if DEBUG
                    for(int k=0; k<SizeBeta; k++) {
                        Debug.Assert(Interaction_n_k[k].Last() == LastInteraction(k), $"{SampleCount}, {k}: interaction not match, {Interaction_n_k[k].Last()}, {LastInteraction(k)}");
                        Debug.Assert(Magnetization_n_k[k].Last() == LastMagnetization(k), $"{SampleCount}, {k}: magnet not match, {Magnetization_n_k[k].Last()}, {LastMagnetization(k)}");
                        Debug.Assert(Energy_n_k[k].Last() == LastEnergy(k), $"{SampleCount}, {k}: energy not mathch,, {Energy_n_k[k].Last()}, {LastEnergy(k)}");
                        //Console.WriteLine($"{n}: OK.");
                    }
                    Console.WriteLine($"{SampleCount}: OK.");
#endif
                }
            }
        }

        private void SamplingLocal() {
            for(int k = 0; k < SizeBeta; k++) {
                for(int r = 0; r < SizeWidth * SizeHeight; r++) {
                    int w = rnd.Next(SizeWidth), h = rnd.Next(SizeHeight);
                    int old_spin = spin_h_w_k[k, w, h];
                    int new_spin = -old_spin;
                    int new_magnetization = oldMagnetization_k[k] - old_spin * 2;
                    int d_interaction = 0;
                    for(int t = 0; t < 4; t++) {
                        int ww = w + dx[t], hh = h + dy[t];
                        if(0 <= ww && ww < SizeWidth && 0 <= hh && hh < SizeHeight) {
                            d_interaction += old_spin * spin_h_w_k[k, ww, hh];
                        }
                    }
                    int new_interaction = oldInteraction_k[k] - d_interaction * 2;
                    float new_energy = Energy(J, Field, new_interaction, new_magnetization);

                    int bi = BetaIndex_n_k[k].Last();
                    if(MetropolisTest(rnd, -Beta_k[bi] * oldEnergy_k[k], -Beta_k[bi] * new_energy)) {
                        spin_h_w_k[k, w, h] = new_spin;
                        oldInteraction_k[k] = new_interaction;
                        oldMagnetization_k[k] = new_magnetization;
                        oldEnergy_k[k] = new_energy;
                    }

                }
                interaction_n_k[k].Add(oldInteraction_k[k]);
                magnetization_n_k[k].Add(oldMagnetization_k[k]);
                energy_n_k[k].Add(oldEnergy_k[k]);
                BetaIndex_n_k[k].Add(BetaIndex_n_k[k].Last());
                ChainIndex_n_k[k].Add(ChainIndex_n_k[k].Last());
            }
        }

        private void SamplingExchange(bool even) {
            int[] chain_index_k = new int[SizeBeta];
            for(int k = 0; k < SizeBeta; k++) {
                chain_index_k[k] = ChainIndex_n_k[k].Last();
            }
            for(int k = (even ? 0 : 1); k < SizeBeta - 1; k += 2) {
                int ciL = chain_index_k[k], ciR = chain_index_k[k + 1];
                if(ExchangeTest(rnd, Beta_k[ciL], Beta_k[ciR], oldEnergy_k[ciL], oldEnergy_k[ciR])) {
                //if(MetropolisTest(rnd, oldLogLikelihood, newLogLikelihood)) {
                        (chain_index_k[k], chain_index_k[k + 1]) = (ciR, ciL);
                }
            }
            for(int k = 0; k < SizeBeta; k++) {
                interaction_n_k[k].Add(oldInteraction_k[k]);
                magnetization_n_k[k].Add(oldMagnetization_k[k]);
                energy_n_k[k].Add(oldEnergy_k[k]);
                ChainIndex_n_k[k].Add(chain_index_k[k]);
                BetaIndex_n_k[chain_index_k[k]].Add(k);
            }
        }

        public void Verbose() {
            var kth_beta = Enumerable.Range(0, SizeBeta).Select(k => BetaIndex_n_k[k].Last());
            Debug.Assert(kth_beta.Count() == SizeBeta);
            Console.WriteLine($"{SampleCount}-th samples");
#if DEBUG
            Console.Write("kth-i\n\t");
            foreach(int ki in kth_beta) {
                Console.Write($"{ki}\t");
            }
            Console.WriteLine("");
#endif

            Console.Write("Interaction\n\t");
            foreach(int ki in kth_beta) { 
                Console.Write($"{oldInteraction_k[ki]}\t");
            }
            Console.WriteLine();
#if DEBUG
            Console.Write("\t");
            foreach(int ki in kth_beta) {
                Console.Write($"{LastInteraction(ki)}\t");
            }
            Console.WriteLine("");
#endif

            Console.Write("Magnetization\n\t");
            foreach(int ki in kth_beta) {
                Console.Write($"{oldMagnetization_k[ki]}\t");
            }
            Console.WriteLine();
#if DEBUG
            Console.Write("\t");
            foreach(int ki in kth_beta) {
                Console.Write($"{LastMagnetization(ki)}\t");
            }
            Console.WriteLine("");
#endif

            Console.Write("Energy\n\t");
            foreach(int ki in kth_beta) { 
            //for(int k = 0; k < SizeBeta; k++) {
                Console.Write($"{oldEnergy_k[ki]}\t");
            }
            Console.WriteLine();
#if DEBUG
            Console.Write("\t");
            foreach(int ki in kth_beta) { 
            //for(int k = 0; k < SizeBeta; k++) {
                Console.Write($"{LastEnergy(ki)}\t");
            }
            Console.WriteLine("");
#endif

        }

        public List<int>[] GetInteractions() => GetSeries(interaction_n_k, ChainIndex_n_k);

        public List<int>[] GetMagnetization() => GetSeries(magnetization_n_k, ChainIndex_n_k);

        public List<float>[] GetEnergys() => GetSeries(energy_n_k, ChainIndex_n_k);

        public (double[], double[]) SpecificHeat_k(int burnin = 1000, int skip = 10, int n_bootstrap = 100) {
            double[] spe_mean_k = new double[SizeBeta], spe_std_k = new double[SizeBeta];
            var energy_n_k = GetEnergys();
            for(int k = 0; k < SizeBeta; k++) {
                var sub_energys = Slice(energy_n_k[k], burnin, null, skip).ToList();
                int n_samples = sub_energys.Count();
                var bootstrap_spes = Enumerable.Range(0, n_bootstrap)
                    .Select(r => Enumerable.Range(0, n_samples).Select(x => (double)sub_energys[rnd.Next(n_samples)]))
                    .Select(lst => SpecificHeat(lst, Beta_k[k], SizeWidth * SizeHeight));
#if DEBUG
                Console.WriteLine($"{bootstrap_spes.ToList()}");
#endif
                spe_mean_k[k] = bootstrap_spes.Average();
                spe_std_k[k] = Stdev(bootstrap_spes);
            }
            return (spe_mean_k, spe_std_k);
        }

    }
}
