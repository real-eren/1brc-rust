use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    ops::Deref,
    sync::{atomic::AtomicUsize, Mutex},
    thread,
};

use memmap2::MmapOptions;
use rustc_hash::FxHashMap;
use slice_allocator::{Config, SliceAllocator};

/// A Rust implementation of the One Billion Row Challenge,
/// originally posed by Gunnar Morling for Java

mod slice_allocator {
    use core::slice;
    use std::{mem::MaybeUninit, sync::Mutex};

    #[derive(Clone, Copy)]
    pub struct Config {
        pub backing_size: usize,
        pub given_size: usize,
    }

    struct Inner<'a> {
        current_buffer: &'a mut [u8],
        buffer_idx: usize,
        buffers: &'a mut Vec<Box<[u8]>>,
        config: Config,
    }

    impl<'a> Inner<'a> {
        fn alloc(&mut self) -> &'a mut [u8] {
            // just a cheeky bit of lifetime laundering. nothing to see here ;)
            // SAFETY: the returned mut slices are exclusive ranges in the boxed slices held in `slices`
            // exclusive because this allocator never backtracks
            // The references live as long as 'a, so they can't be squirreled away by accident

            if self.current_buffer.is_empty() {
                self.prepare_next_buffer();
            }

            // if the remaining buffer wouldn't be enough for the next request, just give the whole
            // slice
            let slice_len = if self.current_buffer.len() < self.config.given_size * 2 {
                self.current_buffer.len()
            } else {
                self.config.given_size
            };
            let (give, keep) = self.current_buffer.split_at_mut(slice_len);

            assert!(
                give.len() >= self.config.given_size,
                "about to return a slice shorter than requested"
            );
            unsafe {
                self.current_buffer =
                    slice::from_raw_parts_mut::<'a>(keep.as_mut_ptr(), keep.len());
                slice::from_raw_parts_mut::<'a>(give.as_mut_ptr(), give.len())
            }
        }

        /// Modifies internal state to point to a new buffer.
        /// Zeroes and re-uses the next existing buffer if available,
        /// else allocates and zeroes a new buffer.
        #[cold]
        fn prepare_next_buffer(&mut self) {
            // need a new buffer

            if !self.buffers.is_empty() {
                // not the very first alloc from an empty base,
                self.buffer_idx += 1;
            }

            if self.buffer_idx == self.buffers.len() {
                // we've used up all the previously allocated buffers
                let mut newstuff = vec![0u8; self.config.backing_size];
                if newstuff.len() != newstuff.capacity() {
                    // Vec::into_boxed_slice will re-allocate into a smaller buffer if there's spare capacity. Want to avoid that.
                    newstuff
                        .spare_capacity_mut()
                        .fill(MaybeUninit::<u8>::zeroed());
                    unsafe { newstuff.set_len(newstuff.capacity()) };
                }
                self.buffers.push(newstuff.into_boxed_slice());
            } else {
                // we have another pre-existing buffer to use.
                // but we don't know if it was zeroed since its last use.
                self.buffers[self.buffer_idx].fill(0);
            }

            unsafe {
                let next_buffer = &mut self.buffers[self.buffer_idx];
                self.current_buffer =
                    slice::from_raw_parts_mut::<'a>(next_buffer.as_mut_ptr(), next_buffer.len());
            }
        }
    }

    /// A thread-safe allocator of u8 slices stored in a borrowed backing buffer
    /// The borrowed backing buffer is used so that the returned slices can outlive this allocator,
    /// which may not be needed for most of an algorithm, but still be bounded predictably by the
    /// backing `Vec<Box<[u8]>>`
    pub struct SliceAllocator<'a> {
        m: Mutex<Inner<'a>>,
    }

    impl<'a> SliceAllocator<'a> {
        /// Constructs a new instance, retaining the
        /// TODO: should we remove or skip the pre-existing buffers that are smaller than
        /// specified in config?
        pub fn new(buffers: &'a mut Vec<Box<[u8]>>, config: Config) -> Self {
            Self {
                m: Mutex::new(Inner {
                    current_buffer: &mut [],
                    buffer_idx: 0,
                    buffers,
                    config,
                }),
            }
        }

        /// Returns a slice at least as big as config.given_size
        /// the slice may not be initialized to zero, if this was constructed with a pre-populated
        /// vec of buffers.
        pub fn alloc(&self) -> &'a mut [u8] {
            let mut this = self.m.lock().unwrap();
            this.alloc()
        }

        /// TODO: verify that this can't be mis-used - i.e. clearing the buffer while references still
        /// exist
        pub unsafe fn clear(mut self) -> Self {
            let this = self.m.get_mut().unwrap();
            this.buffer_idx = 0;
            this.current_buffer = &mut [];
            self
        }
    }
}

struct Map<'a> {
    big_strings: FxHashMap<&'a [u8], CityStats>,
}

impl<'a> Map<'a> {
    fn new() -> Self {
        let mut big_strings = FxHashMap::default();
        big_strings.reserve(4 * 128);
        Self { big_strings }
    }
}

impl<'a> IntoIterator for Map<'a> {
    type Item = (&'a [u8], CityStats);

    // Not sure how to properly abstract the iterator. I know that this couples Map to HashMap,
    // but... why would a consumer of Map need this associated type to begin with
    type IntoIter = std::collections::hash_map::IntoIter<&'a [u8], CityStats>;

    fn into_iter(self) -> Self::IntoIter {
        self.big_strings.into_iter()
    }
}

#[derive(Clone, Copy)]
struct CityStats {
    count: usize,
    sum: isize,
    min: i16,
    max: i16,
}

impl CityStats {
    fn new(measurement: i16) -> Self {
        Self {
            min: measurement,
            max: measurement,
            count: 1,
            sum: measurement as isize,
        }
    }
}

/// Result of a parsing a well-formed line
struct ParsedRow<'a> {
    name: &'a [u8],
    remainder: &'a [u8],
    measurement: i16,
}

/// Finds and parses the first line in chunk. If successful, returns name, measurement and remainder
fn parse_next_line(chunk: &[u8]) -> Option<Result<ParsedRow, ()>> {
    if chunk.is_empty() {
        return None;
    }
    // "cityname;-12.1\n"
    // find semi
    let Some(semi_pos) = chunk.iter().position(|b| *b == b';') else {
        return Some(Err(()));
    };
    let (name, rest) = chunk.split_at(semi_pos);
    // trim semi from rest (guaranteed to be present)
    let rest = unsafe { rest.get_unchecked(1..) };

    let (mut measurement, remainder) =
        if let Some(newline_pos) = rest.iter().position(|b| *b == b'\n') {
            let (l, r) = rest.split_at(newline_pos);
            (l, &r[1..])
        } else {
            (rest, &[] as &[u8])
        };

    let mut sign = 1i16;
    // parse measurement
    if measurement.first().copied() == Some(b'-') {
        measurement = &measurement[1..];
        sign = -1;
    }

    let m_len = measurement.len();
    if (m_len < 3) | (4 < m_len) {
        return Some(Err(()));
    }

    let ones = (measurement[m_len - 1] - b'0') as i16;
    let tens = (measurement[m_len - 3] - b'0') as i16;
    // if measurement is actually only `a.b`, this points to 'a' again. Safely in-bounds, we just
    // need to ignore it.
    let first_val = (measurement[0] - b'0') as i16;
    let hundreds = first_val * (m_len == 4) as i16;

    if (ones.max(tens).max(hundreds) >= 10) | (measurement[m_len - 2] != b'.') {
        return Some(Err(()));
    };
    let measurement = sign * (ones + tens * 10 + hundreds * 100);

    Some(Ok(ParsedRow {
        name,
        remainder,
        measurement,
    }))
}

/// For each line in `chunk`, records into `map`
/// Assumes `chunk` is already aligned to begin at the start of a line,
/// and end at the end of a line
/// If an error is encountered, it returns the location and
fn parse_chunk_and_record<'mmap, 'allocator, 'b>(
    input: &'mmap [u8],
    map: &'b mut Map<'allocator>,
    slice_allocator: &'allocator SliceAllocator,
    global_input_idx: &AtomicUsize,
) -> Result<(), ()> {
    // TODO: use global_input_idx.
    let mut chunk = input;
    // `current_slice` acts as a bump allocator, whose memory is borrowed from, and preserved by, `slice_allocator`.
    let mut current_slice: &mut [u8] = &mut [];

    while let Some(parse_result) = parse_next_line(chunk) {
        let Ok(ParsedRow {
            name,
            measurement,
            remainder,
        }) = parse_result
        else {
            // TODO: return location of error so that it can be examined outside the hotpath
            return Err(());
        };
        chunk = remainder;

        // record entry

        if let Some(entry) = map.big_strings.get_mut(name) {
            entry.count += 1;
            entry.sum += measurement as isize;
            entry.min = entry.min.min(measurement);
            entry.max = entry.max.max(measurement);
        } else {
            // TODO: additional name validations
            if std::str::from_utf8(name).is_err() {
                return Err(());
            }
            if current_slice.len() < name.len() {
                current_slice = slice_allocator.alloc();
            }
            let (slot, rest) = current_slice.split_at_mut(name.len());
            current_slice = rest;
            slot.copy_from_slice(name);

            map.big_strings.insert(slot, CityStats::new(measurement));
        }
    }

    Ok(())
}

/// Collects min, mean and max for each city, from lines in `input`,
/// attempting to use up to `numCores`
/// Stops parsing some time after an error is encountered, but before any writing output.
/// Returns Err if any invalid input encountered during parsing, else Ok.
fn process<'a>(
    num_cores: usize,
    input: &'a [u8],
    alloc_config: Config,
    out: &mut impl Write,
) -> Result<(), ()> {
    let mut vecs = Vec::new();
    let slice_allocator = SliceAllocator::new(&mut vecs, alloc_config);
    let mut maps = (0..num_cores).map(|_| Map::new()).collect::<Vec<Map>>();
    let processing_error = Mutex::new(None);
    let global_input_idx = AtomicUsize::new(0);

    thread::scope(|s| {
        for map in maps.iter_mut() {
            let sa = &slice_allocator;
            let pe = &processing_error;
            let gii = &global_input_idx;
            s.spawn(move || {
                if let Err(e) = parse_chunk_and_record(input, map, sa, gii) {
                    pe.lock().unwrap().get_or_insert(e);
                };
            });
        }
    }); // scope ends, all threads were joined

    if let Some(e) = processing_error.into_inner().unwrap() {
        return Err(e);
    }

    // maps are now populated with data from chunks of the mmap. Merge them.
    // invalid input would have been caught during parsing, so the code below cannot fail

    let mut sorted_long_results = maps.into_iter().fold(
        BTreeMap::<&'a [u8], CityStats>::new(),
        |mut long_results, map| {
            for (name, stats) in map.big_strings.into_iter() {
                long_results
                    .entry(name)
                    .and_modify(|e| {
                        e.sum += stats.sum;
                        e.min = e.min.min(stats.min);
                        e.max = e.max.max(stats.max);
                        e.count += stats.count;
                    })
                    .or_insert(stats);
            }
            long_results
        },
    );
    // display
    write!(out, "{{").unwrap();
    if let Some((name, v)) = sorted_long_results.pop_first() {
        // SAFETY: names are validated as utf8 during parse stage
        write!(
            out,
            "{}={:.1}/{:.1}/{:.1}",
            unsafe { std::str::from_utf8_unchecked(name) },
            v.min as f32 / 10.,
            v.sum as f64 / 10. / v.count as f64,
            v.max as f32 / 10.
        )
        .unwrap();

        for (name, v) in sorted_long_results {
            write!(
                out,
                ", {}={:.1}/{:.1}/{:.1}",
                unsafe { std::str::from_utf8_unchecked(name) },
                v.min as f32 / 10.,
                v.sum as f64 / 10. / v.count as f64,
                v.max as f32 / 10.
            )
            .unwrap();
        }
    }
    write!(out, "}}").unwrap();

    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    let mut buffered_stdout = BufWriter::with_capacity(2 * 1024 * 1024, std::io::stdout());
    let file = File::open("./measurements")?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let num_cores = if let Ok(n) = thread::available_parallelism() {
        n.get()
    } else {
        eprintln!("couldn't query the available parallelism, going single-threaded");
        1
    };
    let num_cores = 1;
    let alloc_config = Config {
        backing_size: 4 * 64 * 4096,
        given_size: 8 * 2 * 4096,
    };
    let slice: &[u8] = mmap.deref();

    match process(num_cores, slice, alloc_config, &mut buffered_stdout) {
        Ok(_) => {
            writeln!(buffered_stdout).unwrap();
            buffered_stdout.flush().unwrap();
        }
        Err(_) => {
            println!("-");
            eprintln!("encountered error while processing file _")
        }
    };
    Ok(())
}

#[cfg(test)]
mod test {
    use core::panic;
    use std::io::BufWriter;

    use crate::{parse_next_line, process, slice_allocator::Config, ParsedRow};

    #[test]
    fn parse_single_lines() {
        for (input, exp_name, exp_measurement) in [
            ("city;12.3", b"city" as &[u8], 123),
            ("c;-1.0", b"c", -10),
            ("ci;-10.2", b"ci", -102),
            ("cit;9.9", b"cit", 99),
        ] {
            let r = parse_next_line(input.as_bytes());
            let ParsedRow {
                name,
                remainder,
                measurement,
            } = match r {
                Some(Ok(pr)) => pr,
                Some(Err(_)) => panic!("unexpected error during parse_next_line; input=`{input}`"),
                None => panic!("strangely"),
            };
            assert_eq!(exp_name, name, "parsing produced wrong name");
            assert_eq!(
                b"", remainder,
                "for these single line inputs, there should be no remainder"
            );
            assert_eq!(
                exp_measurement, measurement,
                "parsing produced wrong number"
            );
        }
    }

    #[test]
    fn basic() {
        let config = Config {
            backing_size: 64 * 4096,
            given_size: 2 * 4096,
        };
        for (input, exp_out) in [
            ("a;-1.1", "{a=-1.1/-1.1/-1.1}"),
            (
                "a;1.1\nabc;12.3\na;-2.2\nverylongname12345;-11.0\naaverylongname1234;0.0",
                "{a=-2.2/-0.6/1.1, aaverylongname1234=0.0/0.0/0.0, abc=12.3/12.3/12.3, verylongname12345=-11.0/-11.0/-11.0}",
            ),
        ] {
            let mut out = Vec::<u8>::with_capacity(1024);
            let mut buf_out = BufWriter::with_capacity(1024, &mut out);
            let process = process(1, input.as_bytes(), config, &mut buf_out);
            assert!(
                process.is_ok(),
                "shouldn't encounter error, was given valid input: `{input}`"
            );
            drop(buf_out);
            let string = match String::from_utf8(out) {
                Ok(s) => s,
                Err(e) => {
                    panic!("produced non-UTF8 output, `{e}`, from input `{input}`");
                }
            };
            assert_eq!(string, exp_out, "bad output for input: `{input}`");
        }
    }
}
