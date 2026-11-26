import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from WS_Mdl.io import sim


class RunMngSelectionTests(unittest.TestCase):
    def setUp(self):
        self.runlog = pd.DataFrame(
            [
                {'MdlN': 'NBr144', 'Start Status': 'Queued', 'End Status': None},
                {'MdlN': 'NBr145', 'Start Status': 'Queued', 'End Status': 'Failed'},
                {'MdlN': 'NBr146', 'Start Status': 'Running', 'End Status': None},
            ]
        )
        self.pool = MagicMock()
        self.pool.__enter__.return_value = self.pool
        self.pool.map.side_effect = lambda func, args: [(arg[1]['MdlN'], True) for arg in args]

    def run_manager(self, **kwargs):
        with (
            patch.object(sim.os, 'chdir'),
            patch.object(sim.pd, 'read_csv', return_value=self.runlog),
            patch.object(sim, 'Pool', return_value=self.pool),
            patch.object(sim, 'sprint'),
        ):
            sim.RunMng(cores=4, Cct_Sims=1, DAG=False, freeze_env='off', **kwargs)

    def test_only_selected_queued_sims_are_started(self):
        self.run_manager(Sims=['NBr144', 'NBr146'])

        args = self.pool.map.call_args.args[1]
        self.assertEqual([arg[1]['MdlN'] for arg in args], ['NBr144'])

    def test_omitting_selection_keeps_all_queued_sims(self):
        self.run_manager()

        args = self.pool.map.call_args.args[1]
        self.assertEqual([arg[1]['MdlN'] for arg in args], ['NBr144', 'NBr145'])

    def test_comma_separated_selection(self):
        self.run_manager(Sims='NBr144, NBr145')

        args = self.pool.map.call_args.args[1]
        self.assertEqual([arg[1]['MdlN'] for arg in args], ['NBr144', 'NBr145'])


if __name__ == '__main__':
    unittest.main()
