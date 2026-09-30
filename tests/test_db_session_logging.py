import logging

from batterygemma.db.session import migrate


def test_migrate_does_not_leave_the_batterygemma_logger_tree_disabled(tmp_path):
    """Regression test: Alembic applies alembic.ini's [loggers] section via logging.config.fileConfig, which
    defaults to disable_existing_loggers=True - that silently disabled every already-created 'batterygemma.*'
    logger (they're created at module-import time, before migrate() ever runs), so every logger.info()/
    .warning() call anywhere in the app became a silent no-op after the first migrate() call, with no
    exception anywhere. migrate() must re-enable the tree it doesn't control the disabling of."""
    logger = logging.getLogger("batterygemma.annotate.tasks")
    logger.disabled = False  # start from a known-good state regardless of test order

    migrate(f"sqlite:///{tmp_path / 'regression.db'}")

    assert logger.disabled is False
    assert logging.getLogger("batterygemma").disabled is False
