#!/usr/bin/env python3
#
# Copyright (c) 2016, 2019 Thomas Perl <m@thp.io>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
# Repetitive command line creation with your editor of choice

Given a list of file names, this will put the list of files into a text
file, open that with `$EDITOR` (or `$VISUAL` if `$EDITOR` is not set). After
the editor returns, any changed names in the text file will cause a
command (default: `mv`) to be executed with the original and new filename.

This is very useful in carrying out rename operations that are easy to
specify with your text editor, but hard to specify with wildcards.

If you ever wrote something like this (and -- as seen here -- still got
shell quoting wrong, because it's tedious), then this tool is for you:

    for f in *.mp3; do; mv "$f" "$(basename $f .mp3)_foo.mp3"; done

Also, you can use different commands instead of `mv`, and add options
for the input and output parameters (`oggenc <infile> -o <outfile>`):

    rpt -c "oggenc {old} -o {new}" *.wav

If you have a command without `{old}` and `{new}`, `{old} {new}` will be
appended to the command line (`cp <infile> <outfile>`):

    rpt -c cp *.wav

You can put `{old}` and `{new}` wherever you want, in different order
and also multiple times (if needed):

    rpt -c "mpg123 -w {new} {old}" *.mp3

To go all-out silly, you can set the `$EDITOR` variable to something
that will automatically change each input line, for example to make
a backup file of every `.py` file in the current folder:

    env EDITOR="sed -i -e '/^[^#]/ s/$/.bak/'" rpt -c cp *.py

This will use `sed` as the editor and in-place edit the temporary
text file created by `rpt` and append `.bak` to all non-comment lines
(the `/^[^#]/` part) in the input file, and with `cp` as the command
will call `cp <filename> <filename>.bak` for every input file.
"""

__author__ = 'Thomas Perl <m@thp.io>'
__license__ = 'Simplified BSD License'
__url__ = 'http://thp.io/2016/rpt/'
__version__ = '2.0.0'


import os
import subprocess
import tempfile
import shlex


class CommandLine(object):
    def __init__(self, command):
        self.command = shlex.split(command)

    def get(self, inputfile, outputfile):
        replace = {
            '{old}': inputfile,
            '{new}': outputfile,
        }

        if all(x in self.command for x in replace.keys()):
            # Have {old} and {new}, replace them in the command line
            return [replace.get(part, part) for part in self.command]
        elif not any(x in self.command for x in replace.keys()):
            # Nothing, append {old} and {new} as parameters to command
            return self.command + [inputfile, outputfile]
        else:
            # Either just {old} or just {new}, this is invalid
            raise ValueError('Command must have both of {old} and {new}, or none, not just one')

    def format(self, inputfile, outputfile):
        return ' '.join(shlex.quote(arg) for arg in self.get(inputfile, outputfile))


def rpt(commandline, filenames, printonly):
    with tempfile.NamedTemporaryFile() as fp:
        fp.write('\n'.join([
            '# For each file that is changed below, this command will be run:',
            '# "{cmd}"'.format(cmd=commandline.format('<oldname>', '<newname>')),
            '# Any unchanged lines will be skipped (no command is run)',
            '# Do not remove, comment out or add any lines (this will fail)',
        ] + filenames).encode('utf-8'))
        fp.flush()

        subprocess.check_call(shlex.split(os.environ.get('EDITOR', os.environ.get('VISUAL', 'vi'))) + [fp.name])
        renames = [line for line in open(fp.name, 'rb').read().decode('utf-8').splitlines()
                   if not line.startswith('#')]

    if len(renames) != len(filenames):
        raise ValueError('Cannot add or remove lines from file')

    # Make sure we don't overwrite anything
    if len(set(renames)) != len(renames):
        raise ValueError('All file names must be unique')

    # TODO: Maybe we want to remove files that have an empty name?
    if any(rename == '' for rename in renames):
        raise ValueError('Empty filenames not supported')

    # Handle cases such as ("a.mp3" -> "b.mp3" and "b.mp3" -> "a.mp3")
    # TODO: Detect that case and use a temporary name to be able to "swap" file contents
    for i, filename in enumerate(filenames):
        for j, rename in enumerate(renames):
            if i != j and filename == rename:
                raise ValueError('Pairwise renames ({filename}) are not yet supported'.format(filename=filename))

    for from_filename, to_filename in zip(filenames, renames):
        if from_filename == to_filename:
            continue

        print(commandline.format(from_filename, to_filename))
        if not printonly:
            subprocess.check_call(commandline.get(from_filename, to_filename))
