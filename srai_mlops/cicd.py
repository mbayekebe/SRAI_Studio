def github_actions_workflow(python_version="3.11",test_command="pytest -q",lint_command="python -m compileall src"):
    return {"name":"ci","on":["push","pull_request"],"jobs":{"test":{"runs-on":"ubuntu-latest","steps":[{"uses":"actions/checkout@v4"},{"uses":"actions/setup-python@v5","with":{"python-version":python_version}},{"run":"pip install -r requirements.txt"},{"run":lint_command},{"run":test_command}]}}}
def deployment_gate(test_passed,coverage,minimum_coverage,security_passed):
    failures=[]
    if not test_passed: failures.append("tests")
    if coverage<minimum_coverage: failures.append("coverage")
    if not security_passed: failures.append("security")
    return {"approved":not failures,"failures":failures}
def semantic_version_bump(version,change):
    major,minor,patch=map(int,version.split("."))
    if change=="major": return f"{major+1}.0.0"
    if change=="minor": return f"{major}.{minor+1}.0"
    return f"{major}.{minor}.{patch+1}"
