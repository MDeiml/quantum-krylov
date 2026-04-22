import subprocess
import git
import numpy as np

repo = git.Repo(search_parent_directories=True)
commit_hash = repo.head.object.hexsha
if repo.is_dirty():
    commit_hash += "_modified"

kappa = 3
num_clusters = 4


def sqlite(sql, output=None) -> str:
    if output is None:
        result = subprocess.run(
            [
                "sqlite3",
                "-cmd",
                ".mode csv",
                "-cmd",
                '.separator ";"',
                "-cmd",
                f".import results/cap_{commit_hash}.csv cap",
                "-cmd",
                f".import results/semi_iterative_{commit_hash}.csv semi_iterative",
                "-cmd",
                sql,
                "-cmd",
                ".quit",
            ],
            capture_output=True,
        )
        if len(result.stderr) > 0:
            print(result.stderr.decode("utf8"))
        return result.stdout.decode("utf8")
    else:
        result = subprocess.run(
            [
                "sqlite3",
                "-cmd",
                ".mode csv --titles on",
                "-cmd",
                '.separator ";"',
                "-cmd",
                f".import results/cap_{commit_hash}.csv cap",
                "-cmd",
                f".import results/semi_iterative_{commit_hash}.csv semi_iterative",
                "-cmd",
                f".output {output}",
                "-cmd",
                sql,
                "-cmd",
                ".quit",
            ],
            capture_output=True,
        )
        if len(result.stderr) > 0:
            print(result.stderr.decode("utf8"))


problems = [
    line.split(";")
    for line in sqlite(
        f"select distinct noise, samples from cap where kappa={kappa} and num_clusters='{num_clusters}' union select distinct noise, samples from semi_iterative where kappa={kappa} and num_clusters='{num_clusters}'"
    ).splitlines()
]

for problem in problems:
    problem_name = "_".join(problem)
    params = [
        line.split(";")
        for line in sqlite(
            f"select distinct transform, adaptive from cap where cap.kappa={kappa} and cap.num_clusters='{num_clusters}'"
        ).splitlines()
    ]

    select = "select c.steps"
    table = " from (select distinct steps from cap union select distinct steps from semi_iterative) as c"
    for i, p in enumerate(params):
        column_name = "_".join(["cap"] + p)
        select += f", a{i}.complexity as {column_name}_complexity"
        select += f", a{i}.'error 5 percentile' as {column_name}_5"
        select += f", a{i}.'error 50 percentile' as {column_name}_50"
        select += f", a{i}.'error 95 percentile' as {column_name}_95"
        table += f" left join cap as a{i} on a{i}.steps = c.steps and a{i}.transform='{p[0]}' and a{i}.adaptive='{p[1]}' and a{i}.noise={problem[0]} and a{i}.samples={problem[1]} and a{i}.kappa={kappa} and a{i}.num_clusters='{num_clusters}'"

    params = [
        line.split(";")
        for line in sqlite(
            "select distinct poly_kind, transform from semi_iterative"
        ).splitlines()
    ]

    for i, p in enumerate(params):
        column_name = "_".join(p)
        select += f", n{i}.complexity as {column_name}_complexity"
        select += f", n{i}.'error 5 percentile' as {column_name}_5"
        select += f", n{i}.'error 50 percentile' as {column_name}_50"
        select += f", n{i}.'error 95 percentile' as {column_name}_95"
        table += f" left join semi_iterative as n{i} on n{i}.steps = c.steps and n{i}.poly_kind='{p[0]}' and n{i}.transform='{p[1]}' and n{i}.noise={problem[0]} and n{i}.samples={problem[1]} and n{i}.kappa={kappa} and n{i}.num_clusters='{num_clusters}'"

    sqlite(select + table, f"processed_results/{problem_name}.csv")

noises = [0, 0.0025, 0.005]
# noises = [0.01, 0.02, 0.04]
samples = [10000, 40000, 160000]
clusters = [num_clusters]
solvers = {
    "$\\qsvt$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='None' and poly_kind='qsvt'",
    "$\\cheb$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='None' and poly_kind='cheb'",
    "$\\qcheb$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='None' and poly_kind='q_cheb'",
    "$\\cups$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='None' and adaptive='False'",
    "$\\caps$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='None' and adaptive='True'",
    "$\\qsvt_\\mathrm{sq}$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='square' and poly_kind='qsvt'",
    "$\\cheb_\\mathrm{sq}$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='square' and poly_kind='cheb'",
    "$\\qcheb_\\mathrm{sq}$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='square' and poly_kind='q_cheb'",
    "$\\cups_\\mathrm{sq}$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='square' and adaptive='False'",
    "$\\caps_\\mathrm{sq}$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='square' and adaptive='True'",
    "$\\qsvt_\\mathrm{sq}'$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='square_outer' and poly_kind='qsvt'",
    "$\\qcheb_\\mathrm{sq}'$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='square_outer' and poly_kind='q_cheb'",
    "$\\cups_\\mathrm{sq}'$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='square_outer' and adaptive='False'",
    "$\\caps_\\mathrm{sq}'$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='square_outer' and adaptive='True'",
}
# noises = [0, 0.005, 0.01, 0.02]
# samples = [160000]
# clusters = [None, 4]
# solvers = {
#     "$\\qsvt$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='None' and poly_kind='qsvt'",
#     "$\\qcheb$": "select steps, semi_iterative.'error 50 percentile', noise, samples, kappa, num_clusters from semi_iterative where transform='None' and poly_kind='q_cheb'",
#     "$\\cups$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='None' and adaptive='False'",
#     "$\\caps$": "select steps, cap.'error 50 percentile', noise, samples, kappa, num_clusters from cap where transform='None' and adaptive='True'",
#     "$\\qcheb_\\mathrm{sq}'$": "select steps, semi_iterative.'error 50 percentile', 2 * noise as noise, samples, kappa, num_clusters from semi_iterative where transform='square_outer' and poly_kind='q_cheb'",
#     "$\\cups_\\mathrm{sq}$": "select steps, cap.'error 50 percentile', 2 * noise as noise, samples, kappa, num_clusters from cap where transform='square' and adaptive='False'",
#     "$\\caps_\\mathrm{sq}$": "select steps, cap.'error 50 percentile', 2 * noise as noise, samples, kappa, num_clusters from cap where transform='square' and adaptive='True'",
# }

errors = np.zeros((len(solvers), len(clusters), len(noises), len(samples)), dtype=float)
errors[:] = np.inf
steps = np.zeros_like(errors, dtype=int)
steps[:] = -1

output = ""

for i, name in enumerate(solvers.keys()):
    for l, c in enumerate(clusters):
        for j, n in enumerate(noises):
            for k, s in enumerate(samples):
                sql = f"select min(t.'error 50 percentile'), steps from ({solvers[name]}) as t where noise={n} and samples={s} and kappa={kappa} and num_clusters='{c}'"
                result = sqlite(sql)
                [error, step] = result.strip().split(";")
                try:
                    errors[i, l, j, k] = error
                    steps[i, l, j, k] = step
                except ValueError:
                    pass

for i, name in enumerate(solvers.keys()):
    output += name
    for l, c in enumerate(clusters):
        for j, n in enumerate(noises):
            for k, s in enumerate(samples):
                error = errors[i, l, j, k]
                step = steps[i, l, j, k]
                if step == -1:
                    output += " & --"
                    continue
                best = i == np.argmin(errors[:, l, j, k])
                error = f"{float(error):.3f}"
                if best:
                    output += f" & \\textbf{{{error}({step})}}"
                else:
                    output += f" & {error}({step})"
    output += " \\\\\n"

print(output)
