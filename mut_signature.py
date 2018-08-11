#!/usr/bin/env python3

import argparse
import matplotlib as mpl
mpl.use("PS")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import sys
import os
from collections import Counter
from itertools import product, combinations, chain
from multiprocessing import Pool
from scipy.misc import comb
from scipy.optimize import nnls, minimize


def main():
    parser = argparse.ArgumentParser(description='Mutational signature analysis')
    parser.add_argument('-r', '--ref', metavar='FILE', required=True, 
                        help='Reference fasta file.')
    parser.add_argument('-s', '--sig', metavar='FILE', default=None,
                        help='Known signature list file. [None]')
    parser.add_argument('-n', '--max-sig-n', metavar='INT', type=int, default=7,
                        help='Max N of known signatures for decomposition. [7]')
    parser.add_argument('-c', '--min-sig-coef', metavar='FLOAT', type=float, default=.1,
                        help='''Coefficient threshold per each signature 
                        for each combination to be included 
                        in signature composition file [0.1]''')
    parser.add_argument('-m', '--min_sig_coef_sum', metavar='FLOAT', type=float, default=.8,
                        help='''Signature coefficient sum threshold 
                        for each combination to be included 
                        in signature composition file [0.8]''')
    parser.add_argument('-j', '--jobs', metavar='INT', default=20,
                        help='''
                        Number of parallel jobs. 
                        If the number of CPU cores is less than default, 
                        it will use 1 less than N of CPU cores. [20]''')
    parser.add_argument('-d', '--dir', metavar='DIR', default='.',
                        help='Output directory. [Current directory]')
    parser.add_argument('-o', '--out', metavar='STR', default='Signature_analysis',
                        help='Prefix for output files. [Signature_analysis]')
    parser.add_argument('infile', nargs='?', default=sys.stdin,
                        type=argparse.FileType('rt'),
                        help='''
                        SNV list file.
                        It should include CHROM, POS, REF, and ALT 
                        as the first 4 columns for generic format.
                        If this is omitted, default is STDIN.''')
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    m = MutSig(args.ref, args.jobs)
    m.read_snv(args.infile)

    if not os.path.exists(args.dir):
       os.makedirs(args.dir)

    if args.sig is not None:
        m.read_sig(args.sig)
        m.decomp(args.max_sig_n)
        m.save_sig_comp("{}/{}.sig_comp.txt".format(args.dir, args.out), 
                        args.min_sig_coef, args.min_sig_coef_sum)
        m.plot_sig_corr_bar("{}/{}.sig_corr_bar.pdf".format(args.dir, args.out))
        m.save_sig_corr("{}/{}.sig_corr.txt".format(args.dir, args.out))
        m.plot_combi2_corr_dist("{}/{}.combi2_corr_dist.pdf".format(args.dir, args.out))
        m.save_combi2_corr("{}/{}.combi2_corr.txt".format(args.dir, args.out))

    m.plot_mut7_spec("{}/{}.mut7_spec.pdf".format(args.dir, args.out))
    m.save_mut7("{}/{}.mut7.txt".format(args.dir, args.out))
    m.plot_mut96_spec("{}/{}.mut96_spec.pdf".format(args.dir, args.out))
    m.save_mut96("{}/{}.mut96.txt".format(args.dir, args.out))

class MutSig:
    def __init__(self, ref_fa, ncore):
        self.ref_fa = ref_fa
        self.ncore = ncore
        self.mut96 = pd.DataFrame(columns=("sub_type", "trinuc", "mut_type", "num", "frac"))
        i = 0
        for r, c in product("CT", "ACGT"):
            if r == c: continue
            for b1, b2 in product("ACGT", repeat=2):
                sub_type = "{}>{}".format(r, c)
                trinuc = "{}{}{}".format(b1, r, b2)
                mut_type = "{}[{}]{}".format(b1, sub_type, b2)
                if mut_type[1:] == "[C>T]G":
                    sub_type = "CpG>TpG"
                self.mut96.loc[i] = [sub_type, trinuc, mut_type, 0, 0]
                i += 1
        self.sig = None

    def trinuc(self, chrom, pos):
        res = subprocess.run(["samtools", "faidx", self.ref_fa, 
                              "{}:{}-{}".format(chrom, int(pos)-1, int(pos)+1)], 
                             stdout=subprocess.PIPE).stdout.decode().splitlines()[1]
        return res

    def revcom(self, nuc):
        return nuc[::-1].translate(str.maketrans("ACGT", "TGCA"))

    def snv2mut_type(self, line):
        chrom, pos, ref, alt = line.split()[:4]
        trinuc = self.trinuc(chrom, pos)

        if ref == "A" or ref == "G":
            sub_type = "{}>{}".format(self.revcom(ref), self.revcom(alt))
            trinuc = self.revcom(trinuc)
        else:
            sub_type = "{}>{}".format(ref, alt)

        mut_type = "{}[{}]{}".format(trinuc[0], sub_type, trinuc[2])
        return mut_type

    def read_snv(self, f):
        with Pool(self.ncore) as pool:
            mut_types = []
            for snv_n, m in enumerate(pool.imap_unordered(self.snv2mut_type, f), 1): 
                mut_types.append(m)
                sys.stderr.write(
                    "\rExtracting signature from snv file: {:>5,} snvs done.".format(snv_n))
            sys.stderr.write("\n")

        for m, n in Counter(mut_types).items():
            self.mut96.loc[self.mut96["mut_type"] == m, "num"] = n
            self.mut96.loc[self.mut96["mut_type"] == m, "frac"] = n / snv_n

        self.mut96["num"] = self.mut96["num"].astype(int)
        self.mut96["frac"] = self.mut96["frac"].astype(float)
        self.mut7 = self.mut96.groupby("sub_type").sum().reset_index()

    def read_sig(self, fname):
        self.sig = pd.read_table(fname, index_col=2)
        self.sig = self.sig.loc[self.mut96["mut_type"]].reset_index().iloc[:,3:]

    def decomp_(self, sig_combi):
        k = len(sig_combi)
        n = len(self.mut96["frac"])
        A, b = self.sig.loc[:,sig_combi], self.mut96["frac"]
        x0, rnorm = nnls(A,b)

        #Define minimisation function
        def fn(x, A, b):
                return np.linalg.norm(A.dot(x) - b)

        #Define constraints and bounds
        #cons = {'type': 'ineq', 'fun': lambda x: 1-np.sum(x)}
        cons = {'type': 'eq', 'fun': lambda x: 1-np.sum(x)}
        bounds = [[0, None] for _ in range(k)]

        #Call minimisation subject to these values
        minout = minimize(fn, x0, args=(A, b), method='SLSQP', bounds=bounds, constraints=cons)
        param = minout.x
        residual = minout.fun**2
        aic = np.round(n*np.log(residual/n) + 2*k, 4)
        pearson_corr = np.round(
            self.sig.loc[:,sig_combi].dot(param).corr(self.mut96["frac"]), 4)
        sig_combi_str = "[{}]".format(
            ", ".join([sig.replace("Signature ", "") for sig in sig_combi]))
        ratio_str = "[{}]".format(
            ", ".join(["{:.4f}".format(r) for r in param]))
        return (k, sig_combi_str, ratio_str, aic, pearson_corr)

    def decomp(self, max_decomp_num):
        with Pool(self.ncore) as pool:
            solutions = []
            combi_num = int(sum(comb(len(self.sig.columns), 
                                     list(range(1, max_decomp_num + 1)))))
            sig_combis = chain.from_iterable(combinations(self.sig, i) 
                                             for i in range(1, max_decomp_num + 1))
            for i, s in enumerate(pool.imap_unordered(self.decomp_, sig_combis), 1):
                solutions.append(s)
                sys.stderr.write("\rDecomposing signature: {0:.2%} done.".format(i/combi_num))
            sys.stderr.write("\n")

        self.sig_comp = pd.DataFrame(
            solutions, columns=("combi_n", "sig_combi", "ratio", "aic", "pearson_corr")
        ).sort_values("aic")

        self.combi2_corr = self.sig_comp.ix[
            self.sig_comp["combi_n"] == 2, ("sig_combi", "ratio", "pearson_corr")
        ].sort_values("pearson_corr", ascending=False)

        self.sig_corr = self.sig_comp.ix[
            self.sig_comp["combi_n"] == 1, ("sig_combi", "pearson_corr")
        ].sort_values("pearson_corr", ascending=False)


    def plot_mut7_spec(self, fname):
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(np.arange(len(self.mut7))+.85, self.mut7["frac"], 
                   width=.3, color=sns.xkcd_rgb["dusty blue"])
            ax.set_xlim((0.4,7.6))
            ax.xaxis.grid(False)
            ax.tick_params(labelsize=12)
            ax.set_xticks(np.arange(1.15,8))
            ax.set_xticklabels(self.mut7["sub_type"], 
                               rotation=45, ha="right", family="Monospace")
            fig.set_tight_layout(True)
            fig.savefig(fname)
            sys.stderr.write("{} saved.\n".format(fname))

    def plot_mut96_spec(self, fname):
        barcolor_mut96 = (
            [sns.xkcd_rgb['blue' ]]*16 + [sns.xkcd_rgb['black']]*16 +
            [sns.xkcd_rgb['red'  ]]*16 + [sns.xkcd_rgb['grey' ]]*16 +
            [sns.xkcd_rgb['green']]*16 + [sns.xkcd_rgb['pink' ]]*16)

        def change_color_alpha(color_hex, alpha):
            r = int(color_hex[1:3], 16)*alpha + 255*(1-alpha)
            g = int(color_hex[3:5], 16)*alpha + 255*(1-alpha)
            b = int(color_hex[5:7], 16)*alpha + 255*(1-alpha)
            return "#{:02x}{:02x}{:02x}".format( int(r + 0.5), int(g + 0.5), int(b + 0.5))

        with sns.axes_style("whitegrid"):
            fig = plt.figure(figsize=(12,3.5))
            ax = fig.add_axes((0.05, 0.12, 0.93, 0.74))

            if self.sig is None:
                ax.bar(np.arange(96.) + 0.5, self.mut96["frac"], linewidth=0, color=barcolor_mut96)
            else:    
                sig_corr = self.sig.corrwith(self.mut96["frac"]).sort_values(ascending=False)
                barcolor_sig = [change_color_alpha(chex, 0.4) for chex in barcolor_mut96]
                ax.bar(np.arange(96.) + 0.5, self.mut96["frac"], 
                       linewidth=0, width=0.3, color=barcolor_mut96)
                ax.bar(np.arange(96.) + 0.9, self.sig[sig_corr.index[0]], 
                       linewidth=0, width=0.3, color=barcolor_sig)

            ax.set_xlim((0,96.6))
            ax.xaxis.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.tick_params(axis='y', labelsize=12)

            ymin, ymax = ax.get_ylim()
            yspan = (ymax - ymin) * 0.02
            ypos = ymax + yspan
            ywidth = yspan * 3 

            for i in range(6):
                ax.add_patch(
                    patches.Rectangle((0.4 + i*16, ypos), 15.9, ywidth, 
                                      clip_on=False, color=barcolor_mut96[0 + i*16]))
                ax.text(6.6 + i*16, ypos + 4.25*yspan,
                        self.mut96["sub_type"][0 + i*16], family="Monospace")

            for xpos, xlabel in enumerate(self.mut96["trinuc"]): 
                ax.text(xpos+0.48 , -6.2*yspan, xlabel[0], 
                        rotation=90, size=10, family="Monospace", 
                        color=change_color_alpha("#000000", 0.7)) 
                ax.text(xpos+0.48 , -4.4*yspan, xlabel[1], 
                        rotation=90, size=10, family="Monospace", 
                        color=barcolor_mut96[xpos], weight="bold") 
                ax.text(xpos+0.48 , -2.6*yspan, xlabel[2], 
                        rotation=90, size=10, family="Monospace", 
                        color=change_color_alpha("#000000", 0.7)) 

            if self.sig is not None: 
                ax.text(79, ymax - 4*yspan, 
                        "{} (R = {:.3f})".format(sig_corr.index[0], sig_corr[0]), 
                        color=barcolor_sig[17], weight="bold") 

            fig.savefig(fname)
            sys.stderr.write("{} saved.\n".format(fname))

    def plot_sig_corr_bar(self, fname):
        def barcolor(idx = None):
            color = [sns.xkcd_rgb['light blue']] * len(self.sig_corr)
            if idx is not None:
                for i in idx: color[i] = sns.xkcd_rgb['pumpkin']
            return color

        sig_names = [name[1:-1] for name in self.sig_corr["sig_combi"]]

        top_pair_index = [
            sig_names.index(name) 
            for name in list(self.combi2_corr["sig_combi"])[0][1:-1].split(", ")]

        fig, ax = plt.subplots()
        ax.barh(np.arange(len(self.sig_corr)-1.,-1,-1) + 0.45, 
                self.sig_corr["pearson_corr"], color=barcolor(top_pair_index))
        ax.yaxis.grid(False)
        ax.set_ylim(0,len(self.sig_corr)-.1)
        ax.set_xticks(np.arange(-0.1,1,0.2))
        ax.set_xticklabels(np.arange(-0.1,1,0.2))
        ax.set_yticks(np.arange(len(self.sig_corr)-1.,-1,-1) + 0.45)
        ax.set_yticklabels(sig_names, family="Monospace")
        ax.tick_params(axis="both", labelsize=12)
        fig.savefig(fname)
        sys.stderr.write("{} saved.\n".format(fname))

    def plot_combi2_corr_dist(self, fname):
        fig, ax = plt.subplots()
        ax.hist(self.combi2_corr["pearson_corr"], bins=66)
        ax.tick_params(labelsize=12)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, 1)
        fig.savefig(fname)
        sys.stderr.write("{} saved.\n".format(fname))

    def save_sig_comp(self, fname, min_sig_coef, min_sig_coef_sum):
        constraint = [
            all(np.array(eval(ratio)) >= min_sig_coef) and 
            sum(np.array(eval(ratio))) >= min_sig_coef_sum 
            for ratio in list(self.sig_comp["ratio"])]
        self.sig_comp[constraint].to_csv(fname, sep="\t", index=False)
        sys.stderr.write("{} saved.\n".format(fname))

    def save_sig_corr(self, fname):
        self.sig_corr.to_csv(fname, sep="\t", index=False)
        sys.stderr.write("{} saved.\n".format(fname))

    def save_combi2_corr(self, fname):
        self.combi2_corr.to_csv(fname, sep="\t", index=False)
        sys.stderr.write("{} saved.\n".format(fname))

    def save_mut7(self, fname):
        self.mut7.to_csv(fname, sep="\t", index=False)
        sys.stderr.write("{} saved.\n".format(fname))

    def save_mut96(self, fname):
        self.mut96.to_csv(fname, sep="\t", index=False)
        sys.stderr.write("{} saved.\n".format(fname))

if __name__ == "__main__":
    main()
