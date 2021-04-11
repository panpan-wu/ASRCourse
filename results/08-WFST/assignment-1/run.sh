#!/bin/bash

fstcompile --isymbols=a.isyms.txt --osymbols=a.osyms.txt --keep_isymbols --keep_osymbols a.fst.txt a.fst
fstcompile --isymbols=b.isyms.txt --osymbols=b.osyms.txt --keep_isymbols --keep_osymbols b.fst.txt b.fst

fstcompose a.fst b.fst a.b.comp.fst

fstprint --isymbols=a.isyms.txt --osymbols=b.osyms.txt a.b.comp.fst a.b.comp.fst.txt
cat a.b.comp.fst.txt
