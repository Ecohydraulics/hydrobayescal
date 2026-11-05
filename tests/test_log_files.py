"""Tests for the log files, which are written into the results folder of a calibration."""
import logging

from hydroBayesCal.utils.config_logging import logger, logger_error, logger_warn, setup_file_logging


def _file_handlers(target):
    return [h for h in target.handlers if isinstance(h, logging.FileHandler)]


def test_the_log_files_follow_the_results_folder(tmp_path):
    first, second = tmp_path / "case_a", tmp_path / "case_b"

    setup_file_logging(str(first))
    logger.info("message for case a")
    setup_file_logging(str(first))  # the same folder again changes nothing
    setup_file_logging(str(second))
    logger.info("message for case b")
    logger_warn.warning("warning for case b")
    logger_error.error("error for case b")

    for target in (logger, logger_warn, logger_error):
        assert len(_file_handlers(target)) == 1, "messages would be written twice"
    a, b = (first / "logfile.log").read_text(), (second / "logfile.log").read_text()
    assert "message for case a" in a and "message for case b" not in a
    assert a.count("HydroBayesCal session") == 1
    assert "message for case b" in b
    assert "warning for case b" in (second / "warnings.log").read_text()
    assert "error for case b" in (second / "errors.log").read_text()


def test_a_restart_appends_to_the_log_of_the_runs(tmp_path):
    folder = tmp_path / "results"
    setup_file_logging(str(folder))
    logger.info("initial runs")
    setup_file_logging(str(tmp_path / "elsewhere"))
    setup_file_logging(str(folder))
    logger.info("restart")

    text = (folder / "logfile.log").read_text()
    assert "initial runs" in text and "restart" in text
    assert text.count("HydroBayesCal session") == 2


def test_the_model_writes_its_log_into_its_results_folder(tmp_path):
    from hydroBayesCal.delft3d.control_delft3d import Delft3DModel

    case = tmp_path / "case"
    case.mkdir()
    (case / "river.mdf").write_text("Runid = #river#\n")
    model = Delft3DModel(case_template_dir=str(case), model_dir=str(tmp_path / "model"),
                         res_dir=str(tmp_path / "results"))
    logger.info("logged after the model was built")

    assert "logged after the model was built" in (tmp_path / "results" / "auto-saved-results-HydroBayesCal"
                                                  / "logfile.log").read_text()
    assert model.asr_dir.endswith("auto-saved-results-HydroBayesCal")
