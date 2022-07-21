using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Simulator.Utils;

namespace MCMCSimulator
{
    internal class Program
    {
        static void Main(string[] args) {
            Console.WriteLine("Width, Height, SampleSize, [beta_min=0.35], [beta_max=0.55], [beta_step=0.01], [bootstrap=100]");
            var parameters = (Console.ReadLine() ?? "").Split(' ');
            int width = int.Parse(parameters[0]), height = int.Parse(parameters[1]), sample_size = int.Parse(parameters[2]), bootstrap = 100;
            decimal beta_min = 0.35m, beta_max = 0.55m, beta_step = 0.01m;
            if(parameters.Length > 3) {
                beta_min = decimal.Parse(parameters[3]);
            }
            if(parameters.Length > 4) {
                beta_max = decimal.Parse(parameters[4]);
            }
            if(parameters.Length > 5) {
                beta_step = decimal.Parse(parameters[5]);
            }
            if(parameters.Length > 6) {
                bootstrap = int.Parse(parameters[6]);
            }

            int burnin = sample_size - sample_size / 10, skip = 10;

            /*
            foreach(decimal beta in ARange(beta_min, beta_max, beta_step))
            //for (decimal beta = beta_min; beta <= beta_max; beta += beta_step)
            {
                var sim = new Simulator.Ising2d(width, height, (float)beta);
                sim.Sampling(sample_size);
                var (spe_mean, spe_std) = sim.SpecificHeats(burnin, 10, bootstrap);
                var ll_mean = -(float)beta * Slice(sim.Energys, burnin, null, 10).Average();
                Console.WriteLine($"{beta}, {spe_mean:f3}, {spe_std:f3}, {ll_mean:f3}");
            }
            */

            float[] betas = ARange(beta_min, beta_max, beta_step).Select(x => (float)x).ToArray();
            var sims = new Simulator.Ising2dExchange(width, height, betas, 10);
            sims.Sampling(sample_size, true, sample_size / 10);
            int sample_num = sample_size + 1;
            PrintListList(ARange(bootstrap, null, skip).Select(
                    n => ARange(betas.Length).Select(
                        k => $"{sims.BetaIndex_n_k[k][n]:f0}"
                    )), ", ");

            List<float>[] energy_n_k = sims.GetEnergys();
            Console.WriteLine($"{energy_n_k.Length}, {energy_n_k[0].Count()}");
            PrintListList(ARange(bootstrap, null, skip).Select(
                    n => ARange(betas.Length).Select(
                        k => $"{energy_n_k[k][n]:f0}"
                    )), ", ");

            var (spe_mean_k, spe_std_k) = sims.SpecificHeat_k(burnin, skip, bootstrap);
            for(int k = 0; k < betas.Length; k++) {
                var ll_mean = -sims.Beta_k[k] * Slice(sims.Energy_n_k[k], burnin, null, skip).Average();
                Console.WriteLine($"{sims.Beta_k[k]:f2}, {spe_mean_k[k]:f3}, {spe_std_k[k]:f3}, {ll_mean:f3}");
            }

            /*
            var (spe_mean_k, spe_std_k) = sim.SpecificHeat_k();
            Console.WriteLine($"{spe_mean_k:F3}");
            Console.WriteLine($"{spe_std_k:F3}");
            */

            Console.Write("End...");
            Console.ReadLine();
        }
    }
}
