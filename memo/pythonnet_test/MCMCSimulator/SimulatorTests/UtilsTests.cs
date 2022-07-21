using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

using Simulator;

namespace SimulatorTests
{
    [TestClass]
    public class UtilsTests
    {
        [TestMethod]
        public void EqualsContainerTest() {
            var array1 = new int[] { 1, 2, 3, 4, 5 };
            var array2 = new int[] { 1, 2, 3, 4, 5 };
            var list1 = new List<int>() { 1, 2, 3, 4, 5 };
            var array3 = new int[] { 2, 3, 4, 5, 6 };
            Assert.IsTrue(Utils.Equals(array1, array2));
            Assert.IsTrue(Utils.Equals(array1, list1));
            Assert.IsFalse(Utils.Equals(array1, array3));
        }

        [TestMethod]
        public void ARangeIntTest() {
            Assert.IsTrue(Utils.Equals(new int[] { 0, 1, 2 }, Utils.ARange(3)));
            Assert.IsTrue(Utils.Equals(new int[] { 1, 2, 3 }, Utils.ARange(1, 4)));
            Assert.IsTrue(Utils.Equals(new int[] { 1, 3, 5 }, Utils.ARange(1, 6, 2)));
            Assert.IsTrue(Utils.Equals(new int[] { 1, 3, 5 }, Utils.ARange(1, 7, 2)));
        }

        [TestMethod]
        public void ARangeDecimalTest() {
            Assert.IsTrue(
                Utils.Equals(new decimal[] { 0, 1, 2 },
                Utils.ARange(3m)));
            Assert.IsTrue(
                Utils.Equals(new decimal[] { 1, 2, 3 },
                Utils.ARange(1m, 4m)));
            Assert.IsTrue(
                Utils.Equals(new decimal[] { 1, 3, 5 },
                Utils.ARange(1m, 6m, 2m)));
            Assert.IsTrue(Utils.Equals(
                new decimal[] { 1, 3, 5 },
                Utils.ARange(1m, 7m, 2m)));
            Assert.IsTrue(Utils.Equals(
                new decimal[] { 0.1m, 0.3m, 0.5m },
                Utils.ARange(0.1m, 0.7m, 0.2m)));
        }

        [TestMethod]
        public void SliceTest() {
            int[] test = Enumerable.Range(0, 10).ToArray();
            //Assert.AreEqual(Utils.Slice(test, 0), (new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
            Assert.IsTrue(Utils.Equals(Utils.Slice(test, 0), (new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 })));
            Assert.IsTrue(Utils.Equals(Utils.Slice(test, 3), (new List<int> { 3, 4, 5, 6, 7, 8, 9 })));
            Assert.IsTrue(Utils.Equals(Utils.Slice(test, 2, 9), (new List<int> { 2, 3, 4, 5, 6, 7, 8 })));
            Assert.IsTrue(Utils.Equals(Utils.Slice(test, 2, 9, 3), (new List<int> { 2, 5, 8 })));
            Assert.IsTrue(Utils.Equals(Utils.Slice(test, 2, 9, 3).ToList(), (new List<int> { 2, 5, 8 })));
        }

        [TestMethod]
        public void VarianceTest() {
            Assert.AreEqual(2.0, Utils.Variance(new double[] { 1, 2, 3, 4, 5 }));
        }

        [TestMethod]
        public void StdevTest() {
            Assert.AreEqual(Math.Sqrt(2.0), Utils.Stdev(new double[] { 1, 2, 3, 4, 5 }));
        }
    }
}