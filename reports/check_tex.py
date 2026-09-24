#!/usr/bin/env python3
"""
check_tex.py -- find why LaTeX says "no legal \\end found".

That message means TeX reached the end of the file without executing
\\end{document}.  The usual causes, all checked here:
  1. \\end{document} is missing (file truncated, e.g. copied from a web page);
  2. an environment (comment, verbatim, ...) or \\iffalse is still open,
     so \\end{document} is swallowed;
  3. a brace { is never closed, so everything after it, including
     \\end{document}, became the argument of some command.

Usage:  python3 check_tex.py paper1.tex
"""
import re
import sys


def strip_comment(line):
    # remove an unescaped % and everything after it
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if ch == "%":
            break
        out.append(ch)
        i += 1
    return "".join(out)


def main(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    lines = raw.split("\n")
    print("file: %s  (%d lines, %d bytes)" % (path, len(lines), len(raw)))
    problems = 0

    if "\\end{document}" not in raw:
        print("  [1] \\end{document} is MISSING. The file is probably truncated;"
              " last non-empty line is:")
        for ln in reversed(lines):
            if ln.strip():
                print("      " + ln[:100])
                break
        problems += 1

    depth = 0
    env_stack = []
    if_stack = []
    first_open_line = None
    for n, ln in enumerate(lines, 1):
        body = strip_comment(ln)
        for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                env_stack.append((name, n))
            else:
                if env_stack and env_stack[-1][0] == name:
                    env_stack.pop()
                elif name != "document" or env_stack:
                    print("  [2] line %d: \\end{%s} does not match the open "
                          "environment %s" % (n, name,
                                              env_stack[-1] if env_stack else None))
                    problems += 1
        for m in re.finditer(r"\\(iffalse|iftrue|if[a-zA-Z@]*|fi)\b", body):
            if m.group(1) == "fi":
                if if_stack:
                    if_stack.pop()
            elif m.group(1) in ("iffalse", "iftrue"):
                if_stack.append((m.group(1), n))
        clean = body.replace("\\{", "").replace("\\}", "")
        for ch in clean:
            if ch == "{":
                if depth == 0:
                    first_open_line = n
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    print("  [3] line %d: extra closing brace }" % n)
                    problems += 1
                    depth = 0

    leftover = [e for e in env_stack if e[0] != "document"]
    if leftover:
        for name, n in leftover:
            print("  [2] environment {%s} opened at line %d is never closed"
                  % (name, n))
        problems += 1
    if if_stack:
        for name, n in if_stack:
            print("  [2] \\%s at line %d has no matching \\fi" % (name, n))
        problems += 1
    if depth > 0:
        print("  [3] %d brace(s) never closed; the outermost was opened near "
              "line %s" % (depth, first_open_line))
        problems += 1

    if problems == 0:
        print("  no structural problem found")
    return problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bad = 0
    for p in sys.argv[1:]:
        bad += main(p)
    sys.exit(1 if bad else 0)
