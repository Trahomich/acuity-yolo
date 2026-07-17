#!/usr/bin/env python3
"""Снимает executable-stack бит (PF_X в PT_GNU_STACK) у .so onnxruntime.

ROCm-сборка onnxruntime содержит .so с PT_GNU_STACK=RWE. В контейнере без
прав на mprotect(PROT_EXEC) загрузка падает:
    "cannot enable executable stack as shared object requires"

execstack/prelink убраны из Debian → правим ELF напрямую на чистом Python.
Поддерживает ELF32 и ELF64 (нам нужен 64 под x86_64 образ).
"""
import glob
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1

count = 0
patterns = [
    '/opt/venv/lib/python*/site-packages/onnxruntime/capi/*.so',
    '/opt/venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.so',
]
for pat in patterns:
    for so in glob.glob(pat):
        with open(so, 'r+b') as f:
            magic = f.read(4)
            if magic != b'\x7fELF':
                continue
            ei_class = f.read(1)[0]  # 1=32bit, 2=64bit
            if ei_class == 2:        # ELF64
                f.seek(32); e_phoff = struct.unpack('<Q', f.read(8))[0]
                f.seek(54); e_phentsize, e_phnum = struct.unpack('<HH', f.read(4))
                phdr_fmt = '<IIQQQQQQ'   # type, flags, offset, vaddr, paddr, filesz, memsz, align
                flags_off = 4            # p_flags идёт сразу после p_type
            else:                      # ELF32
                f.seek(28); e_phoff = struct.unpack('<I', f.read(4))[0]
                f.seek(42); e_phentsize, e_phnum = struct.unpack('<HH', f.read(4))
                phdr_fmt = '<IIIIII'    # type, offset, vaddr, paddr, filesz, memsz, flags, align (partial)
                flags_off = 24          # p_flags в конце Elf32_Phdr
            for i in range(e_phnum):
                base = e_phoff + i * e_phentsize
                f.seek(base)
                p_type = struct.unpack('<I', f.read(4))[0]
                if p_type == PT_GNU_STACK:
                    f.seek(base + flags_off)
                    (p_flags,) = struct.unpack('<I', f.read(4))
                    if p_flags & PF_X:
                        f.seek(base + flags_off)
                        f.write(struct.pack('<I', p_flags & ~PF_X))
                        print(f'cleared execstack: {so}', file=sys.stderr)
                        count += 1
                    break
print(f'done: {count} file(s) patched', file=sys.stderr)
