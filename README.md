# Intro
My Rust solution for Gunnar Morling's [One Billion Row Challenge](https://github.com/gunnarmorling/1brc).
In addition to striving for speed, I also wanted to include input validation,
which, as far as I am aware, all of the top-performing solutions skip,
since that is permissible in the rules of the contest.
I gave myself the added challenge of doing full input validation 
as I feel that would have better carry-over into real applications. 
With that in mind, in this project I explored ways to minimize the overhead of validation, 
summarized below.

# Results

## The hardware
CPU: i7 9750h  
Ram: 16GB  

## The input
A thorough analysis would require a variety of inputs with different characteristics, 
particularly the distribution of city names. For this simple project, 
I just used a single 1 billion row file generated with the script `create_measurements2.sh` 
from Morling's repo.

## Disclaimer
See [this paragraph in another project of mine](https://github.com/real-eren/AoC2022-day13-optimization-rs?tab=readme-ov-file#benchmarks-disclaimer)
for a discussion of the difficulties in accurately benchmarking programs.  

### My Rust solution

#### built with `cargo build --release`
```
$ perf stat -dd -r30 -- ./target/release/rust-1brc  ./measurements.txt > /dev/null 

 Performance counter stats for './target/release/rust-1brc ./measurements.txt' (30 runs):

         40,994.29 msec task-clock:u                     #   10.807 CPUs utilized               ( +-  0.18% )
           210,767      page-faults:u                    #    5.141 K/sec                       ( +-  0.00% )
   104,819,244,549      cycles:u                         #    2.557 GHz                         ( +-  0.06% )  (30.74%)
   172,819,345,365      instructions:u                   #    1.65  insn per cycle              ( +-  0.04% )  (38.53%)
    13,143,905,341      branches:u                       #  320.628 M/sec                       ( +-  0.04% )  (38.60%)
        46,390,058      branch-misses:u                  #    0.35% of all branches             ( +-  0.05% )  (38.68%)
    25,087,172,330      L1-dcache-loads:u                #  611.967 M/sec                       ( +-  0.04% )  (38.75%)
     1,470,769,444      L1-dcache-load-misses:u          #    5.86% of all L1-dcache accesses   ( +-  0.08% )  (38.81%)
        30,773,317      LLC-loads:u                      #  750.673 K/sec                       ( +-  5.29% )  (31.00%)
         3,450,637      LLC-load-misses:u                #   11.21% of all L1-icache accesses   ( +-  0.07% )  (30.95%)
           655,710      L1-icache-load-misses:u                                                 ( +-  1.56% )  (30.85%)
    25,071,851,635      dTLB-loads:u                     #  611.594 M/sec                       ( +-  0.05% )  (30.73%)
         3,797,689      dTLB-load-misses:u               #    0.02% of all dTLB cache accesses  ( +-  0.07% )  (30.63%)
             1,502      iTLB-loads:u                     #   36.639 /sec                        ( +- 22.30% )  (30.58%)
             2,848      iTLB-load-misses:u               #  189.61% of all iTLB cache accesses  ( +- 40.46% )  (30.68%)

            3.7933 +- 0.0139 seconds time elapsed  ( +-  0.37% )
```

#### built with `RUSTFLAGS="-C target-cpu=native" cargo build --release`
```
$ perf stat -dd -r30 -- ./target/release/rust-1brc  ./measurements.txt > /dev/null 

 Performance counter stats for './target/release/rust-1brc ./measurements.txt' (30 runs):

         40,240.47 msec task-clock:u                     #   10.875 CPUs utilized               ( +-  0.20% )
           210,755      page-faults:u                    #    5.237 K/sec                       ( +-  0.00% )
   102,797,821,283      cycles:u                         #    2.555 GHz                         ( +-  0.09% )  (30.74%)
   159,206,876,208      instructions:u                   #    1.55  insn per cycle              ( +-  0.06% )  (38.48%)
    13,158,782,316      branches:u                       #  327.004 M/sec                       ( +-  0.05% )  (38.48%)
        47,373,505      branch-misses:u                  #    0.36% of all branches             ( +-  0.12% )  (38.50%)
    25,092,391,261      L1-dcache-loads:u                #  623.561 M/sec                       ( +-  0.05% )  (38.50%)
     1,489,211,361      L1-dcache-load-misses:u          #    5.93% of all L1-dcache accesses   ( +-  0.09% )  (38.54%)
        27,175,272      LLC-loads:u                      #  675.322 K/sec                       ( +-  3.54% )  (30.90%)
         3,440,839      LLC-load-misses:u                #   12.66% of all L1-icache accesses   ( +-  0.06% )  (30.95%)
           778,683      L1-icache-load-misses:u                                                 ( +-  1.45% )  (30.93%)
    25,050,355,008      dTLB-loads:u                     #  622.516 M/sec                       ( +-  0.06% )  (30.88%)
         3,794,135      dTLB-load-misses:u               #    0.02% of all dTLB cache accesses  ( +-  0.06% )  (30.85%)
             9,339      iTLB-loads:u                     #  232.080 /sec                        ( +- 24.30% )  (30.81%)
             7,530      iTLB-load-misses:u               #   80.63% of all iTLB cache accesses  ( +- 55.22% )  (30.77%)

            3.7002 +- 0.0105 seconds time elapsed  ( +-  0.28% )
```

#### Previous + full LTO
```
$ perf stat -dd -r30 -- ./target/release/rust-1brc  ./measurements.txt > /dev/null 

 Performance counter stats for './target/release/rust-1brc ./measurements.txt' (30 runs):

         37,617.86 msec task-clock:u                     #   10.793 CPUs utilized               ( +-  0.34% )
           210,756      page-faults:u                    #    5.603 K/sec                       ( +-  0.00% )
    98,196,212,547      cycles:u                         #    2.610 GHz                         ( +-  0.05% )  (30.73%)
   153,007,923,598      instructions:u                   #    1.56  insn per cycle              ( +-  0.06% )  (38.43%)
    13,175,437,669      branches:u                       #  350.244 M/sec                       ( +-  0.05% )  (38.44%)
        46,273,123      branch-misses:u                  #    0.35% of all branches             ( +-  0.08% )  (38.45%)
    23,443,023,945      L1-dcache-loads:u                #  623.189 M/sec                       ( +-  0.05% )  (38.48%)
     1,481,329,023      L1-dcache-load-misses:u          #    6.32% of all L1-dcache accesses   ( +-  0.07% )  (38.58%)
        27,304,696      LLC-loads:u                      #  725.844 K/sec                       ( +-  4.58% )  (30.97%)
         3,448,004      LLC-load-misses:u                #   12.63% of all L1-icache accesses   ( +-  0.16% )  (30.98%)
           614,528      L1-icache-load-misses:u                                                 ( +-  3.34% )  (30.95%)
    23,396,358,931      dTLB-loads:u                     #  621.948 M/sec                       ( +-  0.06% )  (30.91%)
         3,798,099      dTLB-load-misses:u               #    0.02% of all dTLB cache accesses  ( +-  0.06% )  (30.86%)
               678      iTLB-loads:u                     #   18.023 /sec                        ( +- 12.47% )  (30.81%)
            34,537      iTLB-load-misses:u               # 5093.95% of all iTLB cache accesses  ( +- 56.83% )  (30.76%)

            3.4855 +- 0.0156 seconds time elapsed  ( +-  0.45% )
```

#### Previous + PGO
```
$ perf stat -dd -r30 -- ./target/release/rust-1brc  ./measurements.txt > /dev/null 

 Performance counter stats for './target/release/rust-1brc ./measurements.txt' (30 runs):

         33,726.11 msec task-clock:u                     #   10.691 CPUs utilized               ( +-  0.32% )
           210,757      page-faults:u                    #    6.249 K/sec                       ( +-  0.00% )
    87,040,719,253      cycles:u                         #    2.581 GHz                         ( +-  0.11% )  (30.71%)
   134,191,257,165      instructions:u                   #    1.54  insn per cycle              ( +-  0.07% )  (38.41%)
    13,096,930,961      branches:u                       #  388.332 M/sec                       ( +-  0.06% )  (38.42%)
        46,577,967      branch-misses:u                  #    0.36% of all branches             ( +-  0.08% )  (38.42%)
    17,176,274,134      L1-dcache-loads:u                #  509.287 M/sec                       ( +-  0.05% )  (38.44%)
     1,502,753,786      L1-dcache-load-misses:u          #    8.75% of all L1-dcache accesses   ( +-  0.09% )  (38.54%)
        31,342,677      LLC-loads:u                      #  929.330 K/sec                       ( +-  6.46% )  (31.03%)
         3,445,769      LLC-load-misses:u                #   10.99% of all L1-icache accesses   ( +-  0.09% )  (31.02%)
           806,326      L1-icache-load-misses:u                                                 ( +-  0.74% )  (30.98%)
    17,136,552,723      dTLB-loads:u                     #  508.109 M/sec                       ( +-  0.06% )  (30.93%)
         3,790,106      dTLB-load-misses:u               #    0.02% of all dTLB cache accesses  ( +-  0.05% )  (30.89%)
             1,816      iTLB-loads:u                     #   53.846 /sec                        ( +- 20.20% )  (30.85%)
             1,434      iTLB-load-misses:u               #   78.96% of all iTLB cache accesses  ( +- 19.75% )  (30.79%)

            3.1545 +- 0.0136 seconds time elapsed  ( +-  0.43% )
```

The difference in run-time between the default release build and the 
`target-cpu=native` + LTO + PGO build is only ~18%; clearly, compiler flags 
won't be enough to close the performance gap.

### [Thomas Wuerthinger's Java submission](https://github.com/gunnarmorling/1brc/blob/main/src/main/java/dev/morling/onebrc/CalculateAverage_thomaswue.java)
Java w/ GraalVM, using the build options from the corresponding `prepare_*.sh` script
```
$ perf stat -dd -r30 -- ./target/CalculateAverage_thomaswue_image > /dev/null 

 Performance counter stats for './target/CalculateAverage_thomaswue_image' (30 runs):

         26,355.95 msec task-clock:u                     #   11.650 CPUs utilized               ( +-  0.25% )
           214,246      page-faults:u                    #    8.129 K/sec                       ( +-  0.00% )
    67,346,371,479      cycles:u                         #    2.555 GHz                         ( +-  0.08% )  (30.76%)
   102,862,570,133      instructions:u                   #    1.53  insn per cycle              ( +-  0.06% )  (38.58%)
     7,524,682,791      branches:u                       #  285.502 M/sec                       ( +-  0.06% )  (38.69%)
        30,200,123      branch-misses:u                  #    0.40% of all branches             ( +-  0.08% )  (38.78%)
    23,541,278,258      L1-dcache-loads:u                #  893.206 M/sec                       ( +-  0.06% )  (38.86%)
     1,960,926,032      L1-dcache-load-misses:u          #    8.33% of all L1-dcache accesses   ( +-  0.06% )  (38.84%)
       145,160,838      LLC-loads:u                      #    5.508 M/sec                       ( +-  1.59% )  (30.82%)
        10,433,983      LLC-load-misses:u                #    7.19% of all L1-icache accesses   ( +-  1.03% )  (30.70%)
         1,266,945      L1-icache-load-misses:u                                                 ( +-  1.13% )  (30.78%)
    23,572,979,421      dTLB-loads:u                     #  894.408 M/sec                       ( +-  0.08% )  (30.95%)
         3,903,080      dTLB-load-misses:u               #    0.02% of all dTLB cache accesses  ( +-  0.11% )  (30.93%)
               344      iTLB-loads:u                     #   13.052 /sec                        ( +- 29.33% )  (30.86%)
             2,307      iTLB-load-misses:u               #  670.64% of all iTLB cache accesses  ( +- 35.15% )  (30.80%)

           2.26240 +- 0.00580 seconds time elapsed  ( +-  0.26% )
```

### [Austin Donisan's C submission](https://github.com/austindonisan/1brc/tree/master)
Both executables were compiled with the recommended flags: `-std=c17 -march=native -mtune=native -Ofast`  

#### clang 14, 12 threads
```
$ perf stat -dd -r30 -- ./1brc.clang14 ~/code/1brc/measurements.txt 12 > /dev/null 

 Performance counter stats for './1brc.clang14 /home/eren/code/1brc/measurements.txt 12' (30 runs):

         10,861.14 msec task-clock:u                     #   11.761 CPUs utilized               ( +-  0.56% )
           214,455      page-faults:u                    #   19.745 K/sec                       ( +-  0.00% )
    27,514,807,994      cycles:u                         #    2.533 GHz                         ( +-  0.29% )  (30.52%)
    37,938,128,759      instructions:u                   #    1.38  insn per cycle              ( +-  0.26% )  (38.45%)
     1,366,540,902      branches:u                       #  125.819 M/sec                       ( +-  0.24% )  (38.63%)
         2,580,440      branch-misses:u                  #    0.19% of all branches             ( +-  0.42% )  (38.87%)
    10,879,385,128      L1-dcache-loads:u                #    1.002 G/sec                       ( +-  0.23% )  (39.02%)
     1,854,094,327      L1-dcache-load-misses:u          #   17.04% of all L1-dcache accesses   ( +-  0.27% )  (38.92%)
       376,931,995      LLC-loads:u                      #   34.705 M/sec                       ( +-  0.27% )  (30.91%)
         3,231,482      LLC-load-misses:u                #    0.86% of all L1-icache accesses   ( +-  0.37% )  (30.77%)
           270,842      L1-icache-load-misses:u                                                 ( +-  1.25% )  (30.80%)
    11,375,272,711      dTLB-loads:u                     #    1.047 G/sec                       ( +-  0.21% )  (30.83%)
         5,509,763      dTLB-load-misses:u               #    0.05% of all dTLB cache accesses  ( +-  0.24% )  (30.73%)
               288      iTLB-loads:u                     #   26.517 /sec                        ( +- 15.91% )  (30.57%)
            14,111      iTLB-load-misses:u               # 4899.65% of all iTLB cache accesses  ( +- 75.52% )  (30.42%)

           0.92352 +- 0.00500 seconds time elapsed  ( +-  0.54% )
```

clang17, 12 threads
```
$ perf stat -dd -r30 -- ./1brc.clang17.0.6 ~/code/1brc/measurements.txt 12 > /dev/null 

 Performance counter stats for './1brc.clang17.0.6 /home/eren/code/1brc/measurements.txt 12' (30 runs):

         10,720.57 msec task-clock:u                     #   11.755 CPUs utilized               ( +-  0.49% )
           214,455      page-faults:u                    #   20.004 K/sec                       ( +-  0.00% )
    27,090,225,096      cycles:u                         #    2.527 GHz                         ( +-  0.31% )  (30.32%)
    36,015,888,222      instructions:u                   #    1.33  insn per cycle              ( +-  0.25% )  (38.20%)
     1,244,743,940      branches:u                       #  116.108 M/sec                       ( +-  0.23% )  (38.53%)
         2,548,189      branch-misses:u                  #    0.20% of all branches             ( +-  0.37% )  (38.85%)
    10,378,269,405      L1-dcache-loads:u                #  968.070 M/sec                       ( +-  0.26% )  (39.16%)
     1,842,380,715      L1-dcache-load-misses:u          #   17.75% of all L1-dcache accesses   ( +-  0.26% )  (39.13%)
       373,832,376      LLC-loads:u                      #   34.871 M/sec                       ( +-  0.22% )  (31.23%)
         3,272,214      LLC-load-misses:u                #    0.88% of all L1-icache accesses   ( +-  0.42% )  (31.25%)
           223,584      L1-icache-load-misses:u                                                 ( +-  2.34% )  (31.06%)
    10,866,856,798      dTLB-loads:u                     #    1.014 G/sec                       ( +-  0.28% )  (30.80%)
         5,503,134      dTLB-load-misses:u               #    0.05% of all dTLB cache accesses  ( +-  0.28% )  (30.50%)
               479      iTLB-loads:u                     #   44.680 /sec                        ( +- 25.40% )  (30.17%)
               548      iTLB-load-misses:u               #  114.41% of all iTLB cache accesses  ( +- 16.59% )  (29.99%)

           0.91197 +- 0.00471 seconds time elapsed  ( +-  0.52% )
```

clang17, 1 thread
```
$ perf stat -dd -r30 -- ./1brc.clang17.0.6 ~/code/1brc/measurements.txt 1 > /dev/null 

 Performance counter stats for './1brc.clang17.0.6 /home/eren/code/1brc/measurements.txt 1' (30 runs):

          4,176.06 msec task-clock:u                     #    0.979 CPUs utilized               ( +-  0.57% )
           211,101      page-faults:u                    #   50.550 K/sec                       ( +-  0.00% )
    13,542,291,854      cycles:u                         #    3.243 GHz                         ( +-  0.12% )  (30.62%)
    35,889,170,163      instructions:u                   #    2.65  insn per cycle              ( +-  0.13% )  (38.36%)
     1,255,185,868      branches:u                       #  300.567 M/sec                       ( +-  0.13% )  (38.36%)
         2,486,703      branch-misses:u                  #    0.20% of all branches             ( +-  0.16% )  (38.36%)
    10,728,847,724      L1-dcache-loads:u                #    2.569 G/sec                       ( +-  0.10% )  (38.47%)
     1,529,886,604      L1-dcache-load-misses:u          #   14.26% of all L1-dcache accesses   ( +-  0.10% )  (38.56%)
        22,012,113      LLC-loads:u                      #    5.271 M/sec                       ( +-  0.52% )  (30.82%)
           533,305      LLC-load-misses:u                #    2.42% of all L1-icache accesses   ( +-  0.28% )  (30.82%)
            94,179      L1-icache-load-misses:u                                                 ( +-  2.05% )  (30.82%)
    10,769,127,085      dTLB-loads:u                     #    2.579 G/sec                       ( +-  0.12% )  (30.82%)
         5,395,686      dTLB-load-misses:u               #    0.05% of all dTLB cache accesses  ( +-  0.12% )  (30.82%)
             1,817      iTLB-loads:u                     #  435.099 /sec                        ( +-  9.32% )  (30.82%)
               597      iTLB-load-misses:u               #   32.86% of all iTLB cache accesses  ( +-  6.98% )  (30.71%)

            4.2654 +- 0.0254 seconds time elapsed  ( +-  0.60% )
```

I also tried LTO + PGO for this submission, but it did not improve any of the metrics.

### `wc -l`
For reference, here is a program often used to count the number of lines in a file. 
It's safe to say that merely counting the lines in a file is a strictly easier problem 
than parsing those lines and performing a hashtable lookup.

```
$ perf stat -dd -r5 -- wc -l ~/code/1brc/measurements2.txt

 Performance counter stats for 'wc -l /home/eren/code/1brc/measurements2.txt' (5 runs):

          7,558.10 msec task-clock:u                     #    1.000 CPUs utilized               ( +-  0.73% )
                69      page-faults:u                    #    9.129 /sec                        ( +-  0.65% )
    25,210,648,308      cycles:u                         #    3.336 GHz                         ( +-  0.02% )  (30.68%)
    96,542,559,573      instructions:u                   #    3.83  insn per cycle              ( +-  0.01% )  (38.40%)
    13,799,308,856      branches:u                       #    1.826 G/sec                       ( +-  0.02% )  (38.45%)
         1,690,005      branch-misses:u                  #    0.01% of all branches             ( +-  0.03% )  (38.50%)
    13,800,251,006      L1-dcache-loads:u                #    1.826 G/sec                       ( +-  0.02% )  (38.56%)
        28,173,408      L1-dcache-load-misses:u          #    0.20% of all L1-dcache accesses   ( +-  2.35% )  (38.60%)
           288,497      LLC-loads:u                      #   38.171 K/sec                       ( +-  5.86% )  (30.88%)
             1,424      LLC-load-misses:u                #    0.49% of all L1-icache accesses   ( +- 18.73% )  (30.83%)
           439,400      L1-icache-load-misses:u                                                 ( +- 11.73% )  (30.78%)
    13,790,619,985      dTLB-loads:u                     #    1.825 G/sec                       ( +-  0.04% )  (30.72%)
                72      dTLB-load-misses:u               #    0.00% of all dTLB cache accesses  ( +- 32.87% )  (30.67%)
            14,847      iTLB-loads:u                     #    1.964 K/sec                       ( +- 17.50% )  (30.67%)
               182      iTLB-load-misses:u               #    1.23% of all iTLB cache accesses  ( +- 31.23% )  (30.67%)

            7.5588 +- 0.0551 seconds time elapsed  ( +-  0.73% )
```

Remarkably, Austin's program with a single thread is much faster than `wc -l`. 
The number of branches here indicates that `wc -l` is going byte-by-byte.

# Optimizations
The commit history documents the optimizations and the rationale behind them.  
The performance comparisons are done with a basic release build, 
without fat LTO, PGO or selecting a narrower target-cpu. 
Each run of `perf stat -dd` ran for 10 iterations (`-r10`), 
and the processor frequency was kept between 2.9 - 3.0 GHZ for the ones that use 12 threads.

## first draft (df2d39d)
* only validate new city names  
  Since each city name is seen many times, if we wait to validate the name until 
  after the hashmap lookup fails, the cost of validation (which is linear with the length of the name),
  is almost negligible.
* bump allocator for names  
  Since we will be storing many small strings, 
  using a bump allocator will improve the locality and reduce the fragmentation.
  Also, we know that all the objects have the same lifetime, so there is no need to have them be individual allocations.
* custom float parsing + storing measurements as `i16`s  
  still much slower than the branchless measurement parsing used in many of the Java 1BRC submissions,
  originally by Quan Anh Mai.
* mmap to open file  
  faster for large files already in the page cache
* single-pass parsing / find delimiter during parsing  
  instead of splitting the input into lines then splitting on delimiters,
  the parse function goes left-to-right without backtracking as much

## distinguish cities with short (<= 16) and long names (e339776)
Short names are stored as `[u8; 16]`, padded with zeros, with a separate hashmap. 
This improves locality - slices, `&[u8]`, occupy as much space as `[u8; 16]` but add a layer of indirection. 
For 'realistic' datasets, short city names are overwhelmingly common. 
This is a form of small string optimization.

```rust
loop { // for each line in input
    // parse line, yielding `name` and `measurement` ...
    // record entry
    if name.len() <= 16 {
        let mut n = [0u8; 16];
        unsafe { n.get_unchecked_mut(0..name.len()).copy_from_slice(name) }

        if let Some(entry) = map.small_strings.get_mut(&n) {
            // update entry stats w/ `measurement` ...
        } else {
            // validate name ...
            map.small_strings.insert(n, CityStats::new(measurement));
        }
    } else {
        // handle big strings with the old code ...
    }
}
// move all the short entries to the big map
for (bytes, stats) in map.small_strings.drain() {
    let name_len = bytes.iter().position(|b| *b == 0).unwrap_or(16);
    let name = unsafe { bytes.get_unchecked(..name_len) };
    if current_slice.len() < name_len {
        current_slice = slice_allocator.alloc();
    }
    let (slot, rest) = current_slice.split_at_mut(name_len);
    current_slice = rest;
    slot.copy_from_slice(name);

    map.big_strings.insert(slot, stats);
}
```

| Stat                  |  old  |  new  | change |
|-----------------------|:-----:|:-----:|:------:|
| Runtime               | 31.7s | 27.7s |  -12%  |
| Instructions          | 280B  | 253B  |  -10%  |
| branch-misses         | 1.63B | 1.32B |  -19%  |
| L1-dcache-load-misses | 797M  | 668M  |  -16%  |


## Parallelize (72c458a)
Fairly straightforward. I used half the available parallelism at this point, 
which I later replaced with 100% (use all cores) and an optional CLI argument for selecting a thread count.

A shared atomic integer is used to coordinate the threads; each thread does a `fetch_add`,
then independently aligns their chunk to start and end at the next newlines. 
If a thread detects an error in the input, it exits early and 
advances the shared counter past the end of the input. 
This will naturally cause all the other threads to exit early, 
without needing a special path to periodically check if an error has been raised. 

Even with such a lightly optimized solution, 
`munmap`'s slowness is showing - the CPU utilization was only 5.647 / 6.
The speedup was only 4.22x despite using 6 cores; however, note that the 
number of cycles barely changed - the average frequency went from 4.0GHZ to 3.1GHZ. 
The other metrics are largely unchanged.  



| Stat                  |  old  |  new  | change  |
|-----------------------|:-----:|:-----:|:-------:|
| Runtime               | 27.7s | 6.55s |  4.22x  |
| cycles                | 112B  | 113B  |         |
| Instructions          | 253B  | 256B  |         |
| branch-misses         | 1.32B | 1.38B |         |
| L1-dcache-load-misses | 668M  | 651M  |         |


## unguarded parsing (6752619)
Here, unguarded means that bounds checking is absent in the hot/inner parsing loop. 
The first line and the last few lines (at least 120 bytes) are handled separately.
Because the first and last lines are handled separately, there is no need to special-case the 
start and end of the input. Additionally, it becomes safe to read 16 bytes ahead without bounds checking.

Remember that we can't assume the input is valid - so how do we handle the case where a newline or semicolon is missing from the whole file? 
Each thread simply checks the last ~100 bytes of each chunk for a newline and semicolon. 
If this fails, the input is definitely invalid and the thread reports an error. If it succeeds, 
then we know that an unguarded search for a semicolon or newline within that chunk will always terminate 
before going past the end of the chunk.

I also used SSE2 intrinsics to speed up the function for finding a semicolon, starting from the start of a line. 
Since most city names are 16 bytes or less, this usually takes just 1 iteration.
```rust
unsafe fn find_semicolon_unguarded(chunk: &[u8]) -> usize {
    use std::arch::x86_64::{_mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};
    const SEMI_MASK: &[u8; 16] = &[b';'; 16];

    let mask = _mm_loadu_si128(SEMI_MASK.as_ptr().cast());
    let mut ptr = chunk.as_ptr().cast();
    let packed = loop {
        let vector = _mm_loadu_si128(ptr);
        let masked = _mm_cmpeq_epi8(mask, vector);
        let packed = _mm_movemask_epi8(masked) as u32;
        if packed != 0 {
            break packed;
        }
        ptr = ptr.byte_add(16);
    };

    // lowest bit of packed corresponds to first byte
    // ptr offset + bit idx in packed.
    (ptr.byte_offset_from(chunk.as_ptr())) as usize + packed.trailing_zeros() as usize
}
```

For finding the subsequent newline, I use SWAR (SIMD within a register) because the valid range for a newline 
is 4-6 bytes after a semicolon. This function also returns a tuple of a value and a `present` flag 
to make it easier to use in a branchless (or less branchy) manner - 
the caller can use the (potentially invalid but safe) value and defer the branching. 
If the more idiomatic `Option<usize>` were returned instead, the caller would need to know the 
implementation details to select a 'free' default for `Option::unwrap_or_else`. 
```rust
fn find_byte_in_word(word: u64, byte: u8) -> (usize, bool) {
    let masked = word ^ (0x0101_0101_0101_0101u64.wrapping_mul(byte as u64));
    let masked =
        (masked.wrapping_sub(0x0101_0101_0101_0101u64)) & !masked & 0x8080_8080_8080_8080u64;
    // the high bit of each matching byte would be set, all other bits are 0
    let lowest_idx = (if masked == 0 {
        64 // ctz returns 64 if no set bit. This is actually less work w/ ctz
    } else {
        masked.trailing_zeros()
    }) / 8;
    let has_byte = (masked != 0) as bool;
    debug_assert!(lowest_idx <= 8, "index should be between 0 and 8");
    (lowest_idx as usize, has_byte)
}
```


(old vs new w/ 6 threads)  

| Stat                  |  old  |  new  | change |
|-----------------------|:-----:|:-----:|:------:|
| Runtime               | 6.55s | 5.59s |  -15%  |
| cycles                | 113B  | 103B  |  -9%   |
| Instructions          | 256B  | 227B  |  -11%  |
| branch-misses         | 1.38B | 591M  |  -57%  |
| L1-dcache-load-misses | 651M  | 656M  |        |

In this commit, I also removed the half parallelism. The result is much less efficient but does run faster.  

(new w/ 6 threads vs new w/ 12 threads)

| Stat                  |  old  |  new   | change  |
|-----------------------|:-----:|:------:|:-------:|
| Runtime               | 5.59s | 4.59s  |  -30%   |
| cycles                | 103B  |  151B  |  +47%   |
| Instructions          | 227B  |  227B  |         |
| branch-misses         | 591M  |  636M  |  +11%   |
| L1-dcache-load-misses | 656M  | 1.465B |  2.23x  |


## use SSE2 intrinsics to copy short names
replaced the following 
```rust
let n = [0u8; 16];
let dest = &mut n[..name.len()];
std::ptr::copy_nonoverlapping(name.as_ptr(), dest.as_mut_ptr(), dest.len());
```
which generates a call to `memcpy`,

with these intrinsics
```rust
use std::arch::x86_64::{_mm_and_si128, _mm_loadu_si128, _mm_storeu_si128};
let n = [0u8; 16];
let bits = _mm_loadu_si128(name.as_ptr().cast());
let mask =
    _mm_loadu_si128(NAME_MASKS.as_ptr().byte_add(16 * name.len()).cast());
let masked_name = _mm_and_si128(bits, mask);
_mm_storeu_si128(n.as_mut_ptr().cast(), masked_name);
```

This was perhaps the simplest change, yet yielded a considerable performance improvement. 
It also indicates that even for simple operations like copying bytes into a short array, 
the compiler output must be inspected. This performance bug did not show up on my profilers; 
I instead found it while tinkering in CompilerExplorer.

| Stat                  |  old   |  new   | change |
|-----------------------|:------:|:------:|:------:|
| Runtime               | 4.59s  | 3.72s  |  -19%  |
| cycles                |  151B  |  120B  |  -20%  |
| Instructions          |  227B  |  192B  |  -15%  |
| branches              |  27B   |  16B   |  -41%  |
| branch-misses         |  636M  |  51M   |  -92%  |
| L1-dcache-loads       |  41B   |  35B   |  -15%  |
| L1-dcache-load-misses | 1.465B | 1.480B |        |


## custom hash, fix inlining
The `#[no_mangle]` attribute on `parse_and_record_unguarded_chunked` prevented inlining,
which had a significant overhead. Removing the attribute or adding `#[inline]` undoes that.

Even after using FxHash, a significant portion of the run-time is spent in the hashtable lookups. 
While rolling a custom hashmap would be faster, I got _some_ perf improvement by simply overriding the 
hash function for the short strings, which are the most common type of city name. The hash function 
was created through experimentation - basically a search for the fewest operations needed to get a 
decently low collision rate. Not much time was spent here, since that time would have been better spent 
on a custom hashmap.
```rust
#[derive(Eq, PartialEq)]
struct ShortStr([u8; 16]);
impl Hash for ShortStr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = self.0.as_ptr();
        let word1 = unsafe { ptr.cast::<u64>().read_unaligned() };
        let word2 = unsafe { ptr.byte_add(8).cast::<u64>().read_unaligned() };
        state.write_u64(
            (word1 ^ word2).wrapping_mul(0u64.wrapping_sub(6116868352871097435))
                ^ (word1 >> 32)
                ^ (word2 >> 16),
        );
    }
}
```

| Stat                  |  old  |  new  | change |
|-----------------------|:-----:|:-----:|:------:|
| Runtime               | 3.72s | 3.15s |  -15%  |
| cycles                | 120B  | 102B  |  -15%  |
| Instructions          | 192B  | 165B  |  -14%  |
| branches              |  16B  |  12B  |  -25%  |
| branch-misses         |  51M  |  46M  |  -10%  |
| L1-dcache-loads       |  35B  |  23B  |  -34%  |
| L1-dcache-load-misses | 1.48B | 1.48B |        |


## branchy city stats update
Assuming that cities have multiple entries and that measurements are semi-randomly distributed, 
these branches become predictable.  
```rust
// before
entry.min = entry.min.min(measurement);
entry.max = entry.max.max(measurement);

// after
if measurement < entry.min {
    entry.min = measurement;
}
if measurement > entry.max {
    entry.max = measurement;
}
```

This did not seem to have a significant effect. Until further evidence is shown of 
the branchy updates being superior, it may be safer to stick with the branchless 
version that is not sensitive to the input.

| Stat                  |  old  |  new  | change  |
|-----------------------|:-----:|:-----:|:-------:|
| Runtime               | 3.15s | 3.15s |         |
| cycles                | 102B  | 101B  |         |
| Instructions          | 165B  | 161B  |  -2.4%  |
| branches              |  12B  |  13B  |   +8%   |
| branch-misses         |  46M  |  46M  |         |
| L1-dcache-loads       |  23B  |  22B  |   -4%   |
| L1-dcache-load-misses | 1.48B | 1.48B |         |


# Known flaws
* The IO code currently only supports files that can be opened with `mmap`,
  so a command like `$ ./rust_1brc <(cat file)` will fail.  
* Many of the top solutions used a second process to shift the cost of the `munmap` syscall. On my machine, this accounted for .2 to .4 seconds of the runtime, 
  which is significant compared to the time actually spent processing (1.92 seconds vs 2.14 seconds for Thomas' solution) 
* Some optimizations not pursued include:
  - custom hashmap  
    Standard hashmaps have to support deletions. 
    Since we don't need deletions for this problem, a specialized hashmap can be leaner,
    particularly in its collision handling (can have linear probing without tombstones).
  - Leaking all memory
    This shouldn't have too much of an effect, as the allocations in this program were large and few.
  - interleaving the parsing of multiple lines  
  - spawning another process so that the munmap can happen asynchronously  
  - generally, anything relying on valid input, such as the fast measurement parsing code used in many of the top Java solutions  

