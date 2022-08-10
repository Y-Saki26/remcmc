using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using static Simulator.Utils;

namespace Simulator
{
    public class Ising2d
    {
        public readonly int SizeWidth, SizeHeight;
        public readonly float Beta, J, Field;
        public List<int> Interactions, Magnetizations;
        public List<float> Energys;
        public int SampleCount;

        private readonly int[] dx = { 0, 0, -1, 1 }, dy = { -1, 1, 0, 0 };
        private int[,] spins;
        private int oldInteraction, oldMagnetization;
        private float oldEnergy;
        private Random rnd;

        public Ising2d(int SizeWidth, int SizeHeight, float Beta, float J = 1.0f, float Field = 0.0f) {
            rnd = new Random();
            this.SizeWidth = SizeWidth;
            this.SizeHeight = SizeHeight;
            this.Beta = Beta;
            this.J = J;
            this.Field = Field;
            spins = new int[SizeWidth, SizeHeight];
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    spins[w, h] = rnd.Next(2) * 2 - 1;
                }
            }
            Interactions = new List<int>();
            Interactions.Add(LastInteraction());
            oldInteraction = Interactions.Last();
            Magnetizations = new List<int>();
            Magnetizations.Add(LastMagnetization());
            oldMagnetization = Magnetizations.Last();
            Energys = new List<float>();
            Energys.Add(LastEnergy());
            oldEnergy = Energys.Last();
            SampleCount = 1;


#if DEBUG
            Console.WriteLine($"spin:");
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    Console.Write($"{spins[w, h]}, ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine($"i: {oldInteraction}");
            Console.WriteLine($"m: {oldMagnetization}");
            Console.WriteLine($"e: {oldEnergy}");

            Verbose();
            Debug.Assert(Interactions.Last() == LastInteraction(), $"{SampleCount}: interaction not match: {Interactions.Last()}, {LastInteraction()}");
            Debug.Assert(Magnetizations.Last() == LastMagnetization(), $"{SampleCount}: magnet not match: {Magnetizations.Last()}, {LastMagnetization()}");
            Debug.Assert(Energys.Last() == LastEnergy(), $"{SampleCount}: energy not mathch: {Energys.Last()}, {LastEnergy()}");
            //Console.WriteLine($"{n}: OK.");
            Console.WriteLine($"{SampleCount}: OK.");
#endif
        }

        //private int Spin(int w, int h) => spins[w, h] * 2 - 1;

        public int LastInteraction() {
            int sum = 0;
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    for(int k=0; k < 4; k++) {
                        int ww = w + dx[k], hh = h + dy[k];
                        if(0<=ww && ww<SizeWidth && 0<=hh && hh < SizeHeight) {
                            sum += spins[w, h] * spins[ww, hh];
                        }
                    }
                    
                }
            }
            return sum / 2;
        }

        public int LastMagnetization() {
            int sum = 0;
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    sum += spins[w, h];// Spin(w, h);
                }
            }
            return sum;
        }

        public float LastEnergy() => Energy(J, Field, oldInteraction, oldMagnetization);
        
        public void Sampling(int sample_size, bool verbose = false, int verbose_frequency = 100) {
            for(int n = 0; n < sample_size; n++) {
                SamplingLocal();

                if(verbose && n % verbose_frequency == 0) {
                    Verbose();
#if DEBUG
                    Debug.Assert(Interactions.Last() == LastInteraction(), $"{SampleCount}: interaction not match: {Interactions.Last()}, {LastInteraction()}");
                    Debug.Assert(Magnetizations.Last() == LastMagnetization(), $"{SampleCount}: magnet not match: {Magnetizations.Last()}, {LastMagnetization()}");
                    Debug.Assert(Energys.Last() == LastEnergy(), $"{SampleCount}: energy not mathch: {Energys.Last()}, {LastEnergy()}");
                    Console.WriteLine($"{n}: OK.");
#endif
                }
            }
        }

        private void SamplingLocal() {
            for(int r = 0; r < SizeWidth * SizeHeight; r++) {
                int w = rnd.Next(SizeWidth), h = rnd.Next(SizeHeight);
                int old_spin = spins[w, h];
                int new_spin = -old_spin;
                int new_magnetization = oldMagnetization - old_spin * 2;
                int d_interaction = 0;
                for(int k = 0; k < 4; k++) {
                    int ww = w + dx[k], hh = h + dy[k];
                    if(0 <= ww && ww < SizeWidth && 0 <= hh && hh < SizeHeight) {
                        d_interaction += old_spin * spins[ww, hh];
                    }
                }
                int new_interaction = oldInteraction - d_interaction * 2;
                float new_energy = Energy(J, Field, new_interaction, new_magnetization);

                if(MetropolisTest(rnd, -Beta * oldEnergy, -Beta * new_energy)) {
                    spins[w, h] = new_spin;
                    oldInteraction = new_interaction;
                    oldMagnetization = new_magnetization;
                    oldEnergy = new_energy;
                }
            }
            Interactions.Add(oldInteraction);
            Magnetizations.Add(oldMagnetization);
            Energys.Add(oldEnergy);
        }

        public void Verbose() {
            Console.WriteLine($"e: {oldEnergy}, j: {oldInteraction}, m: {oldMagnetization}");
            for(int w = 0; w < SizeWidth; w++) {
                for(int h = 0; h < SizeHeight; h++) {
                    Console.Write($"{(spins[w, h] + 1) / 2}, ");
                }
                Console.WriteLine();
            }
        }

        public (double, double) SpecificHeats(int burnin = 1000, int skip = 10, int n_bootstrap = 100) {
            int n_samples = (Energys.Count - 1 - burnin) / skip;
            //Console.WriteLine($"{burnin}, {skip}, {Energys.Count}, {n_samples}");
            var sub_energys = Slice(Energys, burnin, null, skip).ToList();//Energys.Where((value, index) => index >= burnin && (index - burnin) % skip == 0).ToList();
            var bootstrap_spes = Enumerable.Range(0, n_bootstrap)
                .Select(r => Enumerable.Range(0, n_samples).Select(x => (double)sub_energys[rnd.Next(n_samples)]))
                .Select(lst => SpecificHeat(lst, Beta, SizeWidth * SizeHeight));
            return (bootstrap_spes.Average(), Stdev(bootstrap_spes));
        }
    }
}
