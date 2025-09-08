from skills import find_skills

ALIASES = {
    "rest": ["REST", "RESTful", "REST API"],
    "postgres": ["PostgreSQL", "Postgres", "PSQL"],
    "react": ["React", "ReactJS", "React.js"],
    "node.js": ["Node.js", "Nodejs", "Node JS"],
    "aws": ["AWS", "Amazon Web Services"],
    "git": ["Git", "GitHub", "GitLab"],
    "c++": ["C++"],
    "c#": ["C#"],
}


def test_git_does_not_match_digital():
    jd_skills = ["git"]
    resume = "I improved digital experiences."
    assert find_skills(resume, jd_skills, ALIASES) == []


def test_cplusplus_csharp_and_node_rest_variants():
    jd_skills = ["c++", "c#", "node.js", "rest"]
    resume = "I code in C++ and C# and build Node.js backends with RESTful APIs."
    assert find_skills(resume, jd_skills, ALIASES) == ["c++", "c#", "node.js", "rest"]


def test_postgres_variants():
    jd_skills = ["postgres"]
    resume = "Handled PostgreSQL migrations and used PSQL; love Postgres."
    assert find_skills(resume, jd_skills, ALIASES) == ["postgres"]


def test_react_variants():
    jd_skills = ["react"]
    resume = "Built UIs in ReactJS and React.js"
    assert find_skills(resume, jd_skills, ALIASES) == ["react"]


def test_aws_phrase():
    jd_skills = ["aws"]
    resume = "Deployed on Amazon Web Services with EC2."
    assert find_skills(resume, jd_skills, ALIASES) == ["aws"]
