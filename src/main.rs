//! A Rust implementation of the One Billion Row Challenge,
//! originally posed by Gunnar Morling for Java
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufWriter, Write},
    ops::Deref,
    ptr,
    sync::{atomic::AtomicUsize, Mutex},
    thread,
};

use memmap2::MmapOptions;
use rustc_hash::FxHashMap;
use slice_allocator::{AllocConfig, SliceAllocator};

/// ```rust
/// |b: &u8| *b == b'\n'
/// ```
fn is_newline(b: &u8) -> bool {
    *b == b'\n'
}

fn main() -> Result<(), std::io::Error> {
    // todo: CLI args
    let mut buffered_stdout = BufWriter::with_capacity(2 * 1024 * 1024, std::io::stdout());
    let file = File::open("./measurements.txt")?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let num_cores = if let Ok(n) = thread::available_parallelism() {
        // todo: CLI option for numcores
        n.get()
    } else {
        eprintln!("couldn't query the available parallelism, going single-threaded");
        1
    };

    let parse_config = ParseConfig {
        num_cores,
        chunk_size: 4 * 1024 * 1024,
    };
    // todo: CLI opts for alloc config
    let alloc_config = AllocConfig {
        backing_size: 64 * 4096,
        given_size: 2 * 4096,
    };
    let slice: &[u8] = mmap.deref();

    // todo: handle multiple inputs
    match process(slice, parse_config, alloc_config, &mut buffered_stdout) {
        Ok(_) => {
            writeln!(buffered_stdout).unwrap();
            buffered_stdout.flush().unwrap();
        }
        Err(_) => {
            // todo: diagnose error
            println!("-");
            eprintln!("encountered error while processing file _")
        }
    };
    Ok(())
}

#[derive(Clone, Copy)]
struct ParseConfig {
    /// Maximum number of cores
    num_cores: usize,
    /// Length of input each thread will examine at a time
    chunk_size: usize,
}

/// Collects min, mean and max for each city, from lines in `input`,
/// attempting to use up to `numCores`
/// Stops parsing some time after an error is encountered, but before any writing output.
/// Returns Err if any invalid input encountered during parsing, else Ok.
fn process<'a>(
    input: &'a [u8],
    parse_config: ParseConfig,
    alloc_config: AllocConfig,
    out: &mut impl Write,
) -> Result<(), ()> {
    // don't set up parallelization for small inputs
    let num_cores = parse_config
        .num_cores
        .min(input.len() / parse_config.chunk_size)
        .max(1);
    // todo: take these in as parameters to re-use data structures for each input
    let mut vecs = Vec::new();
    let slice_allocator = SliceAllocator::new(&mut vecs, alloc_config);
    let mut maps = (0..num_cores).map(|_| Map::new()).collect::<Vec<Map>>();
    let processing_error = Mutex::new(None);
    let mut global_input_idx = AtomicUsize::new(0);

    // do first line, last 120+ bytes to eliminate edge cases from the other iterations
    let (adjusted_start, end_idx) = {
        let map: &mut Map = &mut maps[0];
        let slice_allocator = &slice_allocator;
        // small enough to just do it with the naive algorithm
        if input.len() <= 1024 {
            parse_and_record_simple(input, map, slice_allocator)?;
            (input.len(), input.len())
        } else {
            const FORWARD_SEARCH_LIMIT: usize = 100 + 1 + 5 + 1;
            let first_newline_pos = input[..FORWARD_SEARCH_LIMIT]
                .iter()
                .position(is_newline)
                .ok_or(())?;
            // find start of line before last 120 bytes,
            const END_PADDING_LENGTH: usize = 120;
            let end_start = 1 + input[..input.len() - END_PADDING_LENGTH]
                .iter()
                .rposition(is_newline)
                .ok_or(())?;
            parse_and_record_simple(&input[..first_newline_pos], map, slice_allocator)?;
            parse_and_record_simple(&input[end_start..], map, slice_allocator)?;
            (first_newline_pos + 1, end_start)
        }
    };
    *global_input_idx.get_mut() = adjusted_start;

    thread::scope(|s| {
        // exit early if we already parsed the whole input
        if *global_input_idx.get_mut() >= end_idx {
            return;
        }
        for map in maps.iter_mut() {
            let sa = &slice_allocator;
            let pe = &processing_error;
            let gii = &global_input_idx;
            s.spawn(move || {
                if let Err(e) = parse_and_record_unguarded_chunked(
                    input,
                    end_idx,
                    parse_config.chunk_size,
                    map,
                    sa,
                    gii,
                ) {
                    pe.lock().unwrap().get_or_insert(e);
                    // let other threads know they should stop
                    gii.store(input.len(), std::sync::atomic::Ordering::SeqCst)
                };
            });
        }
    }); // scope ends, all threads were joined
    if let Some(e) = processing_error.into_inner().unwrap() {
        return Err(e);
    }

    // maps are now populated with data from chunks of the mmap. Merge them.
    // invalid input would have been caught during parsing, so the code below cannot fail
    let mut sorted_long_results = BTreeMap::<&'a str, CityStats>::new();
    for map in maps {
        for (name, stats) in map.big_strings {
            sorted_long_results
                // SAFETY: names are validated as utf8 during parse stage
                .entry(unsafe { std::str::from_utf8_unchecked(name) })
                .and_modify(|e| {
                    e.sum += stats.sum;
                    e.min = e.min.min(stats.min);
                    e.max = e.max.max(stats.max);
                    e.count += stats.count;
                })
                .or_insert(stats);
        }
    }

    // display results, ordered by city name. Format: "{first=min/mean/max, second=min/mean/max, etc=min/mean/max}"
    write!(out, "{{").unwrap();
    if let Some((first_name, first_v)) = sorted_long_results.pop_first() {
        write!(
            out,
            "{}={:.1}/{:.1}/{:.1}",
            first_name,
            first_v.min as f32 / 10.,
            first_v.sum as f64 / 10. / first_v.count as f64,
            first_v.max as f32 / 10.
        )
        .unwrap();

        for (name, v) in sorted_long_results {
            write!(
                out,
                ", {}={:.1}/{:.1}/{:.1}",
                name,
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

struct Map<'a> {
    small_strings: FxHashMap<[u8; 16], CityStats>,
    big_strings: FxHashMap<&'a [u8], CityStats>,
}

impl<'a> Map<'a> {
    fn new() -> Self {
        let mut small_strings = FxHashMap::default();
        small_strings.reserve(4 * 128);
        let mut big_strings = FxHashMap::default();
        big_strings.reserve(4 * 128);
        Self {
            small_strings,
            big_strings,
        }
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

/// Naive implementation for the edges of large inputs or
/// very small inputs that aren't worth spinning up threads for
/// performs bounds checks
#[inline(never)]
fn parse_and_record_simple<'mmap, 'allocator, 'b>(
    whole_input: &'mmap [u8],
    map: &'b mut Map<'allocator>,
    slice_allocator: &'allocator SliceAllocator,
) -> Result<(), ()> {
    // `current_slice` acts as a bump allocator, whose memory is borrowed from, and preserved by, `slice_allocator`.
    let mut current_slice: &'allocator mut [u8] = &mut [];
    let mut chunk = whole_input;

    // now parse and records lines from the chunk
    while let Some(parse_result) = parse_next_line_guarded(chunk) {
        let ParsedRow {
            name,
            measurement,
            remainder,
        } = parse_result?;
        chunk = remainder;

        // we do the lookup with the slice of the input, but store a slice in the
        // slice_allocator, so we can't use the entry API
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

/// Result of a parsing a well-formed line
struct ParsedRow<'a> {
    name: &'a [u8],
    remainder: &'a [u8],
    measurement: i16,
}

/// Finds and parses the first line in chunk. If successful, returns name, measurement and remainder
/// Not heavily optimized for speed
fn parse_next_line_guarded(chunk: &[u8]) -> Option<Result<ParsedRow, ()>> {
    if chunk.is_empty() {
        return None;
    }
    // "cityname;-12.1\n"
    let Some(semi_pos) = chunk.iter().position(|b| *b == b';') else {
        return Some(Err(()));
    };
    let (name, rest) = chunk.split_at(semi_pos);
    // trim semi from rest (guaranteed to be present)
    let rest = unsafe { rest.get_unchecked(1..) };

    let (mut measurement, remainder) =
        if let Some(newline_pos) = rest.iter().take(6).position(is_newline) {
            let (l, r) = rest.split_at(newline_pos);
            (l, &r[1..])
        } else {
            // if newline is absent and this wasn't the end of the array,
            // that will be caught during the checks to m_len
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

    let ones = (measurement[m_len - 1].wrapping_sub(b'0')) as i16;
    let tens = (measurement[m_len - 3].wrapping_sub(b'0')) as i16;
    // if measurement is actually only `a.b`, this points to 'a' again. Safely in-bounds, we just
    // need to ignore it.
    let first_val = (measurement[0].wrapping_sub(b'0')) as i16;
    let hundreds = (m_len == 4) as i16 * first_val;

    let v = sign * (ones + tens * 10 + hundreds * 100);
    if (ones.max(tens).max(hundreds) >= 10) | (measurement[m_len - 2] != b'.') {
        return Some(Err(()));
    };

    Some(Ok(ParsedRow {
        name,
        remainder,
        measurement: v,
    }))
}

/// For each line in `chunk`, records into `map`
/// Polls and increments `global_input_offset` for selecting a chunk.
///
/// todo: If an error is encountered, it returns the location
#[no_mangle]
fn parse_and_record_unguarded_chunked<'mmap, 'allocator, 'b>(
    whole_input: &'mmap [u8],
    end_idx: usize,
    chunk_size: usize,
    map: &'b mut Map<'allocator>,
    slice_allocator: &'allocator SliceAllocator,
    global_input_offset: &AtomicUsize,
) -> Result<(), ()> {
    /// Returns index of first newline in `slice`.
    /// Assumes slice contains a newline, otherwise will go out-of-bounds
    #[no_mangle]
    unsafe fn find_next_newline_unguarded(slice: &[u8]) -> Result<usize, ()> {
        // want this code small, don't expect more than ~dozen iterations
        let mut p = slice.as_ptr();
        let mut num_steps_remaining = 100 + 1 + 5;
        while (num_steps_remaining != 0) & (std::ptr::read(p) != b'\n') {
            p = p.byte_add(1);
            num_steps_remaining -= 1;
        }
        if num_steps_remaining == 0 {
            Err(())
        } else {
            debug_assert_eq!(b'\n', ptr::read(p));
            Ok(p as usize - slice.as_ptr() as usize)
        }
    }
    assert!(end_idx < whole_input.len());

    // `current_slice` acts as a bump allocator, whose memory is borrowed from, and preserved by, `slice_allocator`.
    let mut current_slice: &'allocator mut [u8] = &mut [];

    // get and align chunk, then parse and records lines
    loop {
        let initial_chunk_start =
            global_input_offset.fetch_add(chunk_size, std::sync::atomic::Ordering::SeqCst);
        if initial_chunk_start >= end_idx {
            // all of input has been accounted for, this thread has no further parsing work
            break;
        }
        let initial_chunk_end = (initial_chunk_start + chunk_size).min(end_idx);

        // SAFETY: During process(), we set end_idx to be at least 120 bytes before end of input.
        let Some((chunk_start, chunk_end)) = (unsafe {
            find_next_newline_unguarded(whole_input.get_unchecked(initial_chunk_start..))
                .ok()
                .zip(
                    find_next_newline_unguarded(whole_input.get_unchecked(initial_chunk_end..))
                        .ok(),
                )
        }) else {
            return Err(());
        };
        let (chunk_start, chunk_end) = (
            initial_chunk_start + chunk_start,
            initial_chunk_end + chunk_end,
        );
        debug_assert_eq!(
            b'\n', whole_input[chunk_start],
            "start should point to newline"
        );
        debug_assert_eq!(b'\n', whole_input[chunk_end], "end should point to newline");

        let mut remaining_input = unsafe { whole_input.get_unchecked((chunk_start + 1)..) };
        let chunk_end = unsafe { whole_input.get_unchecked(chunk_end..) }.as_ptr();

        // now parse and records lines from the chunk
        while remaining_input.as_ptr() < chunk_end {
            // SAFETY: during alignment, we verified that there's a newline at the end of chunk
            let ParsedRow {
                name,
                measurement,
                remainder,
            } = unsafe { parse_next_line_unguarded(remaining_input)? };
            remaining_input = remainder;

            // record entry, handle short strings (common case) separately for speed
            // we do the lookup with the slice of the input, but store a slice in the
            // slice_allocator, so we can't use the entry API
            if name.len() <= 16 {
                // do short string path
                let mut n = [0u8; 16];
                unsafe {
                    let dest = &mut n[..name.len()];
                    std::ptr::copy_nonoverlapping(name.as_ptr(), dest.as_mut_ptr(), dest.len());
                    debug_assert_eq!(name, &n[..name.len()]);
                }

                if let Some(entry) = map.small_strings.get_mut(&n) {
                    entry.count += 1;
                    entry.sum += measurement as isize;
                    entry.min = entry.min.min(measurement);
                    entry.max = entry.max.max(measurement);
                } else {
                    if std::str::from_utf8(name).is_err() {
                        return Err(());
                    }
                    map.small_strings.insert(n, CityStats::new(measurement));
                }
            } else {
                if let Some(entry) = map.big_strings.get_mut(name) {
                    entry.count += 1;
                    entry.sum += measurement as isize;
                    entry.min = entry.min.min(measurement);
                    entry.max = entry.max.max(measurement);
                } else {
                    // TODO: additional name validations
                    if std::str::from_utf8(name).is_err()
                        || name.contains(&b'\n')
                        || name.contains(&b';')
                    {
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
        }
    }

    // move all the entries for cities with short names into the main map
    map.big_strings.reserve(map.small_strings.len());
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

    Ok(())
}

/// Finds and parses the first line in chunk. If successful, returns name, measurement and remainder
/// Assumes a newline will be found in chunk. Would read past end of `chunk` if that is not the
/// case
#[no_mangle]
unsafe fn parse_next_line_unguarded(chunk: &[u8]) -> Result<ParsedRow, ()> {
    // record errors w/ this w/out branching
    let mut is_valid = true;

    let semi_pos = find_semicolon_unguarded(chunk);
    let (name, rest) = (
        chunk.get_unchecked(..semi_pos),
        chunk.get_unchecked((semi_pos + 1)..),
    );

    // look for newline. it should be in the next 4-6 bytes
    let next_word = rest.as_ptr().cast::<u64>().read_unaligned();
    let (newline_pos, was_present) = find_byte_in_word(next_word, b'\n');
    is_valid &= was_present;
    // we've recorded whether the newline was range, now make sure the index is safe
    // lowest valid value is 3: "1.2\n"
    let newline_pos = newline_pos.max(3);

    let mut measurement = rest.get_unchecked(..newline_pos);
    debug_assert!(!was_present || !measurement.ends_with(&[b'\n']),);
    let remainder = rest.get_unchecked((newline_pos + 1)..);

    let mut sign = 1i16;
    // parse measurement
    if *measurement.get_unchecked(0) == b'-' {
        measurement = measurement.get_unchecked(1..);
        sign = -1;
    }

    let m_len = measurement.len();
    is_valid &= (3..=4).contains(&m_len);

    // good input: (\d)?\d(.)\d
    let ones = measurement.get_unchecked(m_len - 1).wrapping_sub(b'0') as i16;
    // in the case of invalid input, m_len may be as low as 2,
    // so we should do pointer arith instead of underflowing the index
    let tens = (measurement
        .as_ptr()
        .byte_offset(m_len as isize - 3)
        .read()
        .wrapping_sub(b'0')) as i16;
    // if measurement is actually only `a.b`, this points to 'a' again.
    // Safely in-bounds, we just need to ignore it.
    let first_val = (measurement.get_unchecked(0).wrapping_sub(b'0')) as i16;
    let hundreds = (m_len == 4) as i16 * first_val;

    let v = sign * (ones + tens * 10 + hundreds * 100);
    if (!is_valid)
        | (ones.max(tens).max(hundreds) >= 10)
        | (*measurement.get_unchecked(m_len - 2) != b'.')
    {
        return Err(());
    };

    Ok(ParsedRow {
        name,
        remainder,
        measurement: v,
    })
}

/// Returns 0-based index of the first semicolon in `chunk`
/// Assumes a semicolon will be found before going out-of-bounds
/// Optimized for matches in the first 16 bytes
#[cfg(target_arch = "x86_64")]
#[no_mangle]
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
    let idx = (ptr.byte_offset_from(chunk.as_ptr())) as usize + packed.trailing_zeros() as usize;
    debug_assert_eq!(
        chunk[idx], b';',
        "returned idx does not point to a semicolon"
    );
    idx
}

#[inline(always)]
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

/// A small API to provide bump allocation for byte slices
mod slice_allocator {
    use core::slice;
    use std::{
        mem::{self, MaybeUninit},
        sync::Mutex,
    };

    #[derive(Clone, Copy)]
    pub struct AllocConfig {
        /// Size passed to global allocator
        pub backing_size: usize,
        /// Minimum size of slice returned by [SliceAllocator::alloc]
        pub given_size: usize,
    }

    /// A thread-safe allocator of u8 slices stored in a borrowed backing buffer
    /// The borrowed backing buffer is used so that the returned slices can outlive this allocator,
    /// which may not be needed for most of an algorithm, but still be bounded predictably by the
    /// backing `Vec<Box<[u8]>>`
    pub struct SliceAllocator<'a> {
        m: Mutex<Inner<'a>>,
    }

    impl<'a> SliceAllocator<'a> {
        /// Constructs a new instance, retaining the existing `Box<[u8]>`s, if any, in `buffers`
        pub fn new(buffers: &'a mut Vec<Box<[u8]>>, config: AllocConfig) -> Self {
            Self {
                m: Mutex::new(Inner {
                    current_buffer: &mut [],
                    buffer_idx: 0,
                    buffers,
                    config,
                }),
            }
        }

        /// Returns a slice at least as big as `config.given_size`
        /// the slice may not be initialized to zero, if this was constructed with a pre-populated
        /// vec of buffers.
        pub fn alloc(&self) -> &'a mut [u8] {
            self.m.lock().unwrap().alloc()
        }

        /// TODO: verify that this can't be mis-used - i.e. clearing the buffer while references still
        /// exist
        pub unsafe fn _clear(mut self) -> Self {
            let this = self.m.get_mut().unwrap();
            this.buffer_idx = 0;
            this.current_buffer = &mut [];
            self
        }
    }

    /// Non-threadsafe allocator.
    struct Inner<'a> {
        current_buffer: &'a mut [u8],
        buffer_idx: usize,
        buffers: &'a mut Vec<Box<[u8]>>,
        config: AllocConfig,
    }

    impl<'a> Inner<'a> {
        /// Returns a zeroed region at least as large as `backing_size` specified in Config
        fn alloc(&mut self) -> &'a mut [u8] {
            if self.current_buffer.is_empty() {
                self.prepare_next_backing_buffer();
            }

            // if the remaining buffer wouldn't be enough for the next request, just give the whole
            // slice
            let slice_len = if self.current_buffer.len() < self.config.given_size * 2 {
                self.current_buffer.len()
            } else {
                self.config.given_size
            };
            assert!(
                slice_len >= self.config.given_size,
                "about to return a slice shorter than requested"
            );
            let buffer = mem::replace(&mut self.current_buffer, &mut []);
            let (give, keep) = buffer.split_at_mut(slice_len);
            self.current_buffer = keep;
            give
        }

        /// Modifies internal state to point to a new buffer.
        /// Zeroes and re-uses the next existing buffer if available,
        /// else allocates and zeroes a new buffer.
        #[cold]
        fn prepare_next_backing_buffer(&mut self) {
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
                let next_buffer = self.buffers[self.buffer_idx].as_mut();
                self.current_buffer =
                    slice::from_raw_parts_mut::<'a>(next_buffer.as_mut_ptr(), next_buffer.len());
            }
        }
    }
}

#[cfg(test)]
mod test {
    use core::panic;
    use std::io::BufWriter;

    use crate::{
        find_byte_in_word, find_semicolon_unguarded, parse_next_line_guarded,
        parse_next_line_unguarded, process, slice_allocator::AllocConfig, ParseConfig, ParsedRow,
    };

    #[test]
    fn find_byte_in_word_present() {
        for (input, exp_pos) in [
            (0x01, 0),
            (0x0100, 1),
            (0x0100_0000, 3),
            (0x0100_0000_0000, 5),
        ] {
            let v = find_byte_in_word(input, 0x01);
            assert_eq!((exp_pos, true), v, "input {input:x} -> {v:?}");
        }
    }

    #[test]
    fn find_byte_in_word_absent() {
        for input in [0x0, 0xabbaab, 0xab_0000, 0xFFFF] {
            let v = find_byte_in_word(input, 0xaa);
            assert_eq!(false, v.1, "input {input:x} -> {v:?}");
        }
        for input in [0x01, 0x0101, 0xab_0000, 0xFFFF] {
            let v = find_byte_in_word(input, 0x10);
            assert_eq!(false, v.1, "input {input:x} -> {v:?}");
        }
    }

    #[test]
    fn find_semicolon_unguarded_test() {
        for (input, exp_idx) in [(b";123456701234567" as &[u8], 0), (b"0;23456701234567", 1)] {
            assert!(
                input.len() >= 16,
                "invalid input. fn expects at least 16 bytes"
            );
            assert_eq!(exp_idx, unsafe { find_semicolon_unguarded(input) });
        }
    }

    #[test]
    fn parse_unguarded() {
        for (input, exp_name, exp_measurement, exp_remainder) in [
            ("a;12.3\nqwert12345qwert", "a", 123, "qwert12345qwert"),
            ("a@fj39. ;-1.1\nqwert12345", "a@fj39. ", -11, "qwert12345"),
            ("a9;1.1\nqwert12345qwert", "a9", 11, "qwert12345qwert"),
            ("a b;-99.9\nqwert12345qwert", "a b", -999, "qwert12345qwert"),
        ] {
            let Ok(parsed_row) = (unsafe { parse_next_line_unguarded(input.as_bytes()) }) else {
                panic!("")
            };
            assert_eq!(exp_name.as_bytes(), parsed_row.name);
            assert_eq!(exp_measurement, parsed_row.measurement);
            assert_eq!(exp_remainder.as_bytes(), parsed_row.remainder);
        }
    }

    #[test]
    fn parse_guarded_single_lines() {
        for (input, exp_name, exp_measurement) in [
            ("city;12.3", b"city" as &[u8], 123),
            ("c;-1.0", b"c", -10),
            ("ci;-10.2", b"ci", -102),
            ("cit;9.9", b"cit", 99),
        ] {
            let r = parse_next_line_guarded(input.as_bytes());
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
    fn process_short() {
        let parse_config = ParseConfig {
            num_cores: 1,
            chunk_size: 1024,
        };
        let alloc_config = AllocConfig {
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
            let process = process(input.as_bytes(), parse_config, alloc_config, &mut buf_out);
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

    #[test]
    fn process_long_string() {
        let parse_config = ParseConfig {
            num_cores: 1,
            chunk_size: 1024,
        };
        let alloc_config = AllocConfig {
            backing_size: 64 * 4096,
            given_size: 2 * 4096,
        };
        for (idx, (input_base, exp_out)) in [("a;1.1\n", "{a=1.1/1.1/1.1}")].iter().enumerate() {
            let input = input_base.repeat(1000);
            let mut out = Vec::<u8>::with_capacity(1024);
            let mut buf_out = BufWriter::with_capacity(1024, &mut out);
            let process = process(input.as_bytes(), parse_config, alloc_config, &mut buf_out);
            assert!(
                process.is_ok(),
                "shouldn't encounter error, was given valid input #{idx}"
            );
            drop(buf_out);

            let string = match String::from_utf8(out) {
                Ok(s) => s,
                Err(e) => {
                    panic!("produced non-UTF8 output, `{e}`, from input #{idx}");
                }
            };
            assert_eq!(&string, exp_out, "bad output for input #{idx}");
        }
    }
}
