# texture statistics to compute
# statistics roughly match those used in McDermott & Simoncelli 2011 paper
# however, since PCA will be performed on the statistics, all correlations are used instead of a subset

statistics = {"cochleagram": {"marginals": {"stat_list":["mean", "scaled_std", "skew"],
                                            "kwargs":{"marg_dim":-1}},
                              "correlations": {"stat_list": ["env_c"],
                                               "kwargs":{"diagonals":'all'}}},
              "const_q_modbands": {"norm_marginals": {"stat_list":["norm_power"],
                                                      "kwargs":{"marg_dim":-1, 
                                                                "norm_location_list":["cochleagram/marginals/var"]}}},
              "oct_modbands": {"correlations": {"stat_list": ["c1_corr"],
                                                "kwargs":{"diagonals":'all'}}},
              "oct_modbands_analytic": {"correlations": {"stat_list": ["mod_c2"],
                                                         "kwargs":{}}}}

synthesis_statistics = {"subbands": {"marginals": {"stat_list":["var"],
                                     "kwargs": {"marg_dim":-1}}},
                        "cochleagram": {"marginals": {"stat_list":["mean", "scaled_std", "skew"],
                                                      "kwargs":{"marg_dim":-1}},
                                        "correlations": {"stat_list": ["env_c"],
                                                         "kwargs":{"diagonals":'all'}}},
                        "const_q_modbands": {"norm_marginals": {"stat_list":["norm_power"],
                                                                "kwargs":{"marg_dim":-1, 
                                                                          "norm_location_list":["cochleagram/marginals/var"]}}},
                        "oct_modbands": {"correlations": {"stat_list": ["c1_corr"],
                                                          "kwargs":{"diagonals":'all'}}},
                        "oct_modbands_analytic": {"correlations": {"stat_list": ["mod_c2"],
                                                                   "kwargs":{}}}}