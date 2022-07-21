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
        public List<int>[] Interaction_n_k, Magnetization_n_k, BetaIndex_n_k, ChainIndex_n_k;
        public List<float>[] Energy_n_k;

        private readonly int[] dx = { 0, 0, -1, 1 }, dy = { -1, 1, 0, 0 };
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
            this.ExchangeRate = ExchangeStep;
            SizeBeta = Beta_k.Length;
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

            SampleCount = 1;
            Interaction_n_k = new List<int>[SizeBeta];
            Magnetization_n_k = new List<int>[SizeBeta];
            Energy_n_k = new List<float>[SizeBeta];
            BetaIndex_n_k = new List<int>[SizeBeta];
            ChainIndex_n_k = new List<int>[SizeBeta];
            oldInteraction_k = new int[SizeBeta];
            oldMagnetization_k = new int[SizeBeta];
            oldEnergy_k = new float[SizeBeta];
            for(int k = 0; k < SizeBeta; k++) {
                Interaction_n_k[k] = new List<int>();
                Interaction_n_k[k].Add(LatsInteraction(k));
                oldInteraction_k[k] = Interaction_n_k[k].Last();
                Magnetization_n_k[k] = new List<int>();
                Magnetization_n_k[k].Add(LastMagnetization(k));
                oldMagnetization_k[k] = Magnetization_n_k[k].Last();
                Energy_n_k[k] = new List<float>();
                Energy_n_k[k].Add(LastEnergy(k));
                oldEnergy_k[k] = Energy_n_k[k].Last();
                BetaIndex_n_k[k] = new List<int>();
                BetaIndex_n_k[k].Add(k);
                ChainIndex_n_k[k] = new List<int>();
                ChainIndex_n_k[k].Add(k);
            }
        }

        public int LatsInteraction(int k) {
            int sum = 0;
            for(int w = 0; w < SizeWidth - 1; w++) {
                int ww = w + 1;
                for(int h = 0; h < SizeHeight - 1; h++) {
                    int hh = h + 1;
                    sum += spin_h_w_k[k, w, h] * spin_h_w_k[k, ww, hh];
                }
            }
            return sum;
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
                if(n % ExchangeRate != 0) {
                    SamplingLocal();
                } else {
                    SamplingExchange((n / ExchangeRate) % 2 == 0);
                }
                SampleCount++;

                if(verbose && n % verbose_frequency == 0) {
                    Verbose();
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

                    if(MetropolisTest(rnd, -Beta_k[BetaIndex_n_k[k].Last()] * oldEnergy_k[k], -Beta_k[BetaIndex_n_k[k].Last()] * new_energy)) {
                        spin_h_w_k[k, w, h] = new_spin;
                        oldInteraction_k[k] = new_interaction;
                        oldMagnetization_k[k] = new_magnetization;
                        oldEnergy_k[k] = new_energy;
                    }

                }
                Interaction_n_k[k].Add(oldInteraction_k[k]);
                Magnetization_n_k[k].Add(oldMagnetization_k[k]);
                Energy_n_k[k].Add(oldEnergy_k[k]);
                BetaIndex_n_k[k].Add(BetaIndex_n_k[k].Last());
                ChainIndex_n_k[k].Add(ChainIndex_n_k[k].Last());
            }
        }

        private void SamplingExchange(bool even) {
            int[] rev_index_k = new int[SizeBeta];
            for(int k = 0; k < SizeBeta; k++) {
                rev_index_k[k] = ChainIndex_n_k[k].Last();
            }
            for(int k = (even ? 0 : 1); k < SizeBeta - 1; k += 2) {
                int riL = rev_index_k[k], riR = rev_index_k[k + 1];
                float oldLogLikelihood = -Beta_k[riL] * oldEnergy_k[riL] - Beta_k[riR] * oldEnergy_k[riR],
                    newLogLikelihood = -Beta_k[riR] * oldEnergy_k[riR] - Beta_k[riL] * oldEnergy_k[riR];
                if(MetropolisTest(rnd, oldLogLikelihood, newLogLikelihood)) {
                    //(oldInteraction_k[kL], oldInteraction_k[kR]) = (oldInteraction_k[kR], oldInteraction_k[kL]);
                    //(oldMagnetization_k[kL], oldMagnetization_k[kR]) = (oldMagnetization_k[kR], oldMagnetization_k[kL]);
                    //(oldEnergy_k[kL], oldEnergy_k[kR]) = (oldEnergy_k[kR], oldEnergy_k[kL]);
                    (rev_index_k[k], rev_index_k[k + 1]) = (riR, riL);
                }
            }
            for(int k = 0; k < SizeBeta; k++) {
                Interaction_n_k[k].Add(oldInteraction_k[k]);
                Magnetization_n_k[k].Add(oldMagnetization_k[k]);
                Energy_n_k[k].Add(oldEnergy_k[k]);
                ChainIndex_n_k[k].Add(rev_index_k[k]);
                BetaIndex_n_k[rev_index_k[k]].Add(k);
            }
        }

        public void Verbose() {
            var kth_beta = Enumerable.Range(0, SizeBeta).Select(k => BetaIndex_n_k[k].Last()).ToList();
            for(int k = 0; k < SizeBeta; k++) {
                Console.Write($"{oldEnergy_k[kth_beta[k]]}, ");
            }
            Console.WriteLine();
        }

        public List<int>[] GetInteractions() => GetSeries(Interaction_n_k, ChainIndex_n_k);

        public List<int>[] GetMagnetization() => GetSeries(Magnetization_n_k, ChainIndex_n_k);

        public List<float>[] GetEnergys() => GetSeries(Energy_n_k, ChainIndex_n_k);

        public (double[], double[]) SpecificHeat_k(int burnin = 1000, int skip = 10, int n_bootstrap = 100) {
            double[] spe_mean_k = new double[SizeBeta], spe_std_k = new double[SizeBeta];
            var energy_n_k = GetEnergys();
            for(int k = 0; k < SizeBeta; k++) {
                var sub_energys = Slice(energy_n_k[k], burnin, null, skip).ToList();
                int n_samples = sub_energys.Count();
                var bootstrap_spes = Enumerable.Range(0, n_bootstrap)
                    .Select(r => Enumerable.Range(0, n_samples).Select(x => (double)sub_energys[rnd.Next(n_samples)]))
                    .Select(lst => SpecificHeat(lst, Beta_k[k], SizeWidth * SizeHeight));
                Debug.WriteLine($"{bootstrap_spes.ToList()}");
                spe_mean_k[k] = bootstrap_spes.Average();
                spe_std_k[k] = Stdev(bootstrap_spes);
            }
            return (spe_mean_k, spe_std_k);
        }

    }
}
