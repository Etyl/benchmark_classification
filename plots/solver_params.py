from benchopt import BasePlot


class Plot(BasePlot):
    name = "Solver Params"
    type = "bar_chart" # Or "scatter", "boxplot", "table"
    options = {
        "dataset": ..., # Automatic options from DataFrame columns
        "solver": ["SVM", "torch-logreg"], # custom options
        "objective": ...,
        "objective_column": ...,
    }

    # The inputs args of this method correspond to `df` and
    # the keys in the `options` dictionary.
    def plot(self, df, dataset, solver, objective, objective_column):
        # Filter the DataFrame
        df = df.query(
            "dataset_name == @dataset and objective_name == @objective"
        )
        plots = []
        for solver_name, df_filtered in df.groupby('solver_name'):
            if solver not in solver_name:
                continue
            df_filtered = df_filtered.select_dtypes(include=['number'])
            
            if objective_column not in df_filtered:
                continue

            # For each solver, get the last recorded value of the
            # objective metric for each run
            stop_val = df_filtered['stop_val'].max()
            this_df = df_filtered[df_filtered['stop_val'] == stop_val]
            y = this_df[objective_column].tolist()

            # For a bar chart, each plot is a dictionary with keys:
            # 'y', 'text', 'label', 'color'
            # For other plot types, the structure may differ (see documentation).
            plots.append({
                "y": y,
                "text": "",
                "label": solver_name,
                "color": self.get_style(solver_name)["color"]
            })

        return plots

    # The inputs args of this method correspond to `df` and
    # the keys in the `options` dictionary.
    def get_metadata(self, df, dataset, solver, objective, objective_column):
        return {
            "title": f"{objective}\nData: {dataset}\nSolver: {solver}",
            "ylabel": objective_column,
        }