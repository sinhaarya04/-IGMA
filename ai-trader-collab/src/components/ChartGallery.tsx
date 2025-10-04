import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { BarChart3 } from "lucide-react";
import { ChartData } from "./Dashboard";

interface ChartGalleryProps {
  charts: ChartData[];
}

export const ChartGallery = ({ charts }: ChartGalleryProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Chart Gallery
        </CardTitle>
      </CardHeader>
      <CardContent>
        {charts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-muted-foreground">Charts will appear here</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {charts.map((chart) => (
              <div
                key={chart.id}
                className="border border-border rounded-lg p-4 bg-background hover:border-primary/50 transition-colors"
              >
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  {chart.title}
                </h4>
                
                <iframe
                  srcDoc={chart.html}
                  className="w-full h-64 mb-2 border-0"
                  title={chart.title}
                  sandbox="allow-scripts allow-same-origin"
                />
                
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium">Why:</span> {chart.explanation}
                </p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
