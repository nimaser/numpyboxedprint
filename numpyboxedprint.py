import numpy as np
import re

def parr(
    arr                 : np.ndarray,
    max_precision       : int   = 4,
    dot_odds            : bool  = True,
    bracket_inner_1D    : bool  = False,
    box_inner_1D        : bool  = False
) -> None:
    """Pretty prints an np.ndarray in a boxed style inspired by APL's array printing"""
    if not isinstance(arr, np.ndarray): raise TypeError
    
    ### array entry formatting ###
    
    def fmt_entry(parts : tuple[str], maxlens : tuple[int], ljust : tuple[bool]) -> str:
        """Takes parts of an entry (eg integer and decimal parts of a float), the corresponding
        maximum lengths those parts can have, and whether they should be left justified (if not,
        they will be right justified), and combines them into a single aligned entry string"""
        part_data = zip(parts, maxlens, ljust)
        return "".join(part.ljust(ml) if lj else part.rjust(ml) for part, ml, lj in part_data)
    
    def fmt_entries(arr : np.ndarray) -> np.ndarray:
        """Convert entries to equal-length space-padded strings with aligned decimal points/signs"""
        # use numpy's printer to get the minimal string representation of each array entry
        stringified = [np.array2string(np.array(x), precision=max_precision, floatmode="maxprec")
            for x in arr.flat]
        # set which regex patterns will be used to extract the parts of each entry, as well as
        # whether each part of each entry should be left justified
        if np.issubdtype(arr.dtype, np.integer):
            pattern = "([+-]?[0-9]+)"
            ljust = (0,)                     # int part of integer, trivial case
        elif np.issubdtype(arr.dtype, np.floating):
            pattern = "([+-]?[0-9]+.)([0-9]*)(e[+-][0-9]+)?"
            ljust = (0, 1, 1)                # int, dec, exp parts of real float
        elif np.issubdtype(arr.dtype, np.complex128):
            pattern = "([+-]?[0-9]+.)([0-9]*)(e[+-][0-9]+)?([+-][0-9]+.)([0-9]*)(e[+-][0-9]+)?(j)"
            ljust = (0, 1, 1, 0, 1, 1, 0) # int,dec,exp parts of Re,Im with +- in mid, j at end
        else:
            raise NotImplementedError("types beyond int, float, and complex not supported")
        # process entries with regex, so each list elem is a tuple representing a match, and each
        # tuple element is a captured group (part) of the match; then replace None with ""
        regroups = [re.search(pattern, s).groups() for s in stringified]
        regroups = [tuple(e if e is not None else "" for e in t) for t in regroups]
        # consider corresponding groups across matches (i.e. parts) and for each, find the max len
        # of its elements, which all corresponding parts will be padded out to
        maxlens = [max(len(s) for s in g) for g in zip(*regroups)]
        return np.array([fmt_entry(t, maxlens, ljust) for t in regroups]).reshape(arr.shape)
    
    ### array formatting ###
    
    def box(lines : list[str], dotted = False) -> list[str]:
        """Puts a unicode box around some list of string lines"""
        # box characters
        h, v = '─', '│'
        tl, tr = '┌', '┐'
        bl, br = '└', '┘'
        if dotted: h, v = '╌', '┊'
        # a box consists of a top, a bottom, and the original input bordered with pipes
        width = max(map(len, lines))
        top = [f"{tl}{h*width}{tr}"]
        bot = [f"{bl}{h*width}{br}"]
        mid = [f"{v}{l.ljust(width)}{v}" for l in lines] # making sure to pad shorter lines
        return top + mid + bot
        
    def vstack(blocks : list[list[str]]) -> list[str]:
        """Stacks a list of blocks (each a list of equal-length string lines) vertically"""
        lines = [line for b in blocks for line in b]
        width = max(map(len, lines))
        return [l.ljust(width) for l in lines] # making sure to pad shorter lines
        
    def hstack(blocks : list[list[str]], spacing : int = 0) -> list[str]:
        """Stacks a list of blocks (each a list of equal-length string lines) horizontally"""
        # bottom pad the blocks that are shorter to make them the same height,
        # and make sure the added lines are the same length as the block's first line
        height = max(map(len, blocks))
        blocks = [b + [' '*len(b[0])] * (height - len(b)) for b in blocks]
        # extract corresponding lines from each block and concatenate them
        line_tuples = zip(*blocks)
        return [(' '*spacing).join(lt) for lt in line_tuples]
    
    def fmt_arr(arr : np.ndarray) -> list[str]:
        # low-dimensional base cases
        if arr.ndim == 0: return [arr.item()]
        if bracket_inner_1D and arr.ndim == 1: return ['[' + ' '.join(arr) + ']']
        if not box_inner_1D and arr.ndim == 1: return [' '.join(arr)]
        # recursive case: fmt each subarray of arr, then stack the resulting blocks
        fmtted = [fmt_arr(s) for s in arr]
        if arr.ndim % 2 == 0: return box(vstack(fmtted), dotted=False)
        return box(hstack(fmtted), dotted=dot_odds)
    
    if arr.ndim == 1: box_inner_1D = True # so that 1D arrays on their own are still boxed properly
    print('\n'.join(fmt_arr(fmt_entries(arr)))) # gg

### examples ###
    
if __name__ == "__main__":
    parr(np.random.permutation(np.arange(-18, 18)).reshape(6, 3, 2))
    parr((np.random.rand(6, 3, 2)*21)-10)
    parr((np.random.rand(3, 2, 3, 2, 1)*21)-10 + (np.random.rand(3, 2, 3, 2, 1)*21j)-10j)
