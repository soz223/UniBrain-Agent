# Embedded demonstration

Updated November 23, 2026, America/New_York.

Open https://soz223.github.io/UniBrain-Agent/?demo=1 or choose **Try demo**
in the regular application. The original Pipeline and Training views render
embedded simulated API responses using their existing components and animations.
The interface labels these values as simulated. They are not experimental results.

## Classroom walkthrough

1. Open Pipeline Workspace and inspect the default stages and joint objective.
2. Choose an optional stage or replace a component, then apply the changes.
3. Confirm and run to show simulated preparation, updates and model outputs.
4. Open Training to explore the gradient flow animation, task actors, weights,
   experiment outcomes and fold details. Play flow controls the original animation.
5. Exit demo to return to normal backend connection behavior.

The demonstration executes entirely in the browser. It does not contact the
training server, send prompts to an LLM, modify checkpoints, or use GPU time.
The tool selections and numerical trajectories illustrate the interface only.
Unknown simulated routes fail locally instead of falling through to a real API.
Real backend credentials and saved connection settings are separate from demo mode.

## Maintenance

`web/demo.js` supplies the simulated API contract. `web/connection.js` selects
the embedded transport. `unibrain_agent/frontend.py` includes it in the same
document used by the local server and GitHub Pages. `scripts/build_frontend.py`
publishes that document as `index.html`.

Run the frontend tests and exercise Pipeline and Training in a browser before
publishing. Preserve explicit simulation labels when adding examples. Never
populate the public demonstration with private records or real access tokens.

The historical independent inference service is documented in
`docs/classroom-demo.md`; it is not the current classroom entry point.
