import subprocess
import git
import itertools

# repo = git.Repo(search_parent_directories=True)
# commit_hash = repo.head.object.hexsha
# if repo.is_dirty():
#     commit_hash += "_modified"

commit_hash = "59898b8caddb6e95b0c0f004b564ed9ec01607cd"

def sqlite(sql, output=None) -> str:
    if output is None:
        result = subprocess.run(["sqlite3", "-cmd", ".mode csv", "-cmd", '.separator ";"', "-cmd", f".import results/adaptive_{commit_hash}.csv adaptive", "-cmd", f".import results/nonadaptive_{commit_hash}.csv nonadaptive", "-cmd", sql, "-cmd", ".quit"], capture_output=True)
        if len(result.stderr) > 0:
            print(result.stderr.decode("utf8"))
        return result.stdout.decode("utf8")
    else:
        result = subprocess.run(["sqlite3", "-cmd", ".mode csv --titles on", "-cmd", '.separator ";"', "-cmd", f".import results/adaptive_{commit_hash}.csv adaptive", "-cmd", f".import results/nonadaptive_{commit_hash}.csv nonadaptive", "-cmd", f".output {output}", "-cmd", sql, "-cmd", ".quit"], capture_output=True)
        if len(result.stderr) > 0:
            print(result.stderr.decode("utf8"))


problems = [l.split(";") for l in sqlite("select distinct noise, samples from adaptive union select distinct noise, samples from nonadaptive").splitlines()]

for problem in problems:
    problem_name = "_".join(problem)
    params = [l.split(";") for l in sqlite("select distinct sup_norm_constraint, square from adaptive").splitlines()]

    select = "select c.steps"
    table = " from (select distinct steps from adaptive union select distinct steps from nonadaptive) as c"
    for i, p in enumerate(params):
        column_name = "_".join(["adaptive"] + p)
        select += f", a{i}.'error 0 percentile' as {column_name}_0"
        select += f", a{i}.'error 50 percentile' as {column_name}_50"
        select += f", a{i}.'error 100 percentile' as {column_name}_100"
        table += f" left join adaptive as a{i} on a{i}.steps = c.steps and a{i}.sup_norm_constraint='{p[0]}' and a{i}.square='{p[1]}' and a{i}.noise={problem[0]} and a{i}.samples={problem[1]}"

    params = [l.split(";") for l in sqlite("select distinct poly_kind, square from nonadaptive").splitlines()]

    for i, p in enumerate(params):
        column_name = "_".join(p)
        select += f", n{i}.'error 0 percentile' as {column_name}_0"
        select += f", n{i}.'error 50 percentile' as {column_name}_50"
        select += f", n{i}.'error 100 percentile' as {column_name}_100"
        table += f" left join nonadaptive as n{i} on n{i}.steps = c.steps and n{i}.poly_kind='{p[0]}' and n{i}.square='{p[1]}' and n{i}.noise={problem[0]} and n{i}.samples={problem[1]}"

    sqlite(select + table, f"processed_results/{problem_name}.csv")

noises = [0, 0.01, 0.02]
samples = [10000, 40000, 160000]
poly_kind = {
    "qsvt": "$P_\\qsvt$",
    "chebyshev_positive": "$P_\\cheb$",
    "chebyshev_symmetric": "$P_\\chebsym$",
}
sup_norm_constraint = [False, True]

output = ""

for p in poly_kind.keys():
    output += poly_kind[p]
    output += " & $\\times$ & --"
    for n, s in itertools.product(noises, samples):
        sql = f"select min(nonadaptive.'error 50 percentile'), steps from nonadaptive where noise={n} and samples={s} and square='False' and poly_kind='{p}'"
        result = sqlite(sql)
        [error, steps] = result.strip().split(";")
        try:
            error = f"{float(error):.3f}"
        except ValueError:
            pass
        output += f" & {error}({steps})"
    output += " \\\\\n"

for c in sup_norm_constraint:
    output += "adaptive"
    output += " & $\\times$"
    output += " & $\\checkmark$" if c else " & $\\times$"
    for n, s in itertools.product(noises, samples):
        sql = f"select min(adaptive.'error 50 percentile'), steps from adaptive where noise={n} and samples={s} and square='False' and sup_norm_constraint='{c}'"
        result = sqlite(sql)
        [error, steps] = result.strip().split(";")
        try:
            error = f"{float(error):.3f}"
        except ValueError:
            pass
        output += f" & {error}({steps})"
    output += " \\\\\n"

for p in poly_kind.keys():
    output += poly_kind[p]
    output += " & $\\checkmark$ & --"
    for n, s in itertools.product(noises, samples):
        sql = f"select min(nonadaptive.'error 50 percentile'), steps from nonadaptive where noise={n} and samples={s} and square='True' and poly_kind='{p}'"
        result = sqlite(sql)
        [error, steps] = result.strip().split(";")
        try:
            error = f"{float(error):.3f}"
        except ValueError:
            pass
        output += f" & {error}({steps})"
    output += " \\\\\n"

for c in sup_norm_constraint:
    output += "adaptive"
    output += " & $\\checkmark$"
    output += " & $\\checkmark$" if c else " & $\\times$"
    for n, s in itertools.product(noises, samples):
        sql = f"select min(adaptive.'error 50 percentile'), steps from adaptive where noise={n} and samples={s} and square='True' and sup_norm_constraint='{c}'"
        result = sqlite(sql)
        [error, steps] = result.strip().split(";")
        try:
            error = f"{float(error):.3f}"
        except ValueError:
            pass
        output += f" & {error}({steps})"
    output += " \\\\\n"

print(output)
