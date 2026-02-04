from benchopt import BasePlot


class Plot(BasePlot):
    name = "Solver Params"
    type = "bar_chart"
    options = {
        "dataset": ...,
        "solver": ["SVM", "torch-logreg"],
        "objective": ...,
        "objective_column": ...,
    }

    def plot(self, df, dataset, solver, objective, objective_column):
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

            stop_val = df_filtered['stop_val'].max()
            this_df = df_filtered[df_filtered['stop_val'] == stop_val]
            y = this_df[objective_column].tolist()

            plots.append({
                "y": y,
                "text": "",
                "label": solver_name,
                "color": self.get_style(solver_name)["color"]
            })

        return plots

    def get_metadata(self, df, dataset, solver, objective, objective_column):
        return {
            "title": f"{objective}\nData: {dataset}\nSolver: {solver}",
            "ylabel": objective_column,
        }