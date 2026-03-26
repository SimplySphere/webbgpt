from eval.catalog import score_catalog_response


def test_score_catalog_response_handles_expected_course_and_citation():
    exact, has_citation, abstained = score_catalog_response(
        {"expected_course_codes": ["CS 101"]},
        "CS 101 is an introductory programming course. [source: CS 101]",
    )
    assert exact == 1.0
    assert has_citation is True
    assert abstained is False


def test_score_catalog_response_treats_expected_abstention_as_exact():
    exact, has_citation, abstained = score_catalog_response(
        {"expected_course_codes": [], "expects_abstention": True},
        "I could not find a matching catalog entry, so I cannot confirm the prerequisites.",
    )
    assert exact == 1.0
    assert has_citation is False
    assert abstained is True
