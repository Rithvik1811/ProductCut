import type { Metadata } from "next";
import Home from "./components/home/Home";

export const metadata: Metadata = {
  title: "ProductCut — AI Product Video Studio",
  description:
    "ProductCut reads your real product photos and has a team of AI agents script, shoot, and cut a short honest ad for your shop.",
};

export default function Page() {
  return <Home />;
}
